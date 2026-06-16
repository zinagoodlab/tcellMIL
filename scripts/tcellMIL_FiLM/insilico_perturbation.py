#!/usr/bin/env python3
"""
insilico_perturbation.py — in-silico TF overexpression / knockdown using the
checkpoints produced by save_checkpoints.py.

Per fold (= per held-out LOOCV patient under seed 2 of 24c):

  1. Load the autoencoder + MIL checkpoint trained on the OTHER 63 patients.
  2. Read the held-out patient's cell × 154-TF AUCell matrix (already
     shifted to [-1, 1] by `(x - 0.5) * 2`).
  3. Predict baseline `prob_OR` using the patient's cohort one-hot (FiLM
     input). This must match `predictions.csv` from the checkpoint run
     to within float precision -- if it doesn't, the checkpoints are
     mismatched.
  4. For each TF i in 0..153:
       - Per-cell additive 3-MAD perturbation, both directions:
           up:   cells[:, i] = clip(cells[:, i] + 3*MAD_i, -1, 1)
           down: cells[:, i] = clip(cells[:, i] - 3*MAD_i, -1, 1)
         where MAD_i is computed across the patient's own cells for that
         TF (matches the original framework, see
         /Users/kristintsui/HA_MIL_model/Perturbation_experiment_per_cell.ipynb).
       - Re-encode the perturbed cells with the SAME frozen AE.
       - Forward through the SAME frozen MIL with the SAME cohort one-hot.
       - Record `delta = prob_OR_perturbed - prob_OR_base`.

FiLM detail: cohort one-hot is fixed per patient (perturbation does not
change cohort membership). It must be passed every forward call -- if
omitted, FiLM modulation falls through and the model behaves like the
no-FiLM baseline, not 24c.

Output: long-form CSV `perturbation_long.csv` with one row per
(patient, TF, direction).
"""

import argparse
import gc
import glob
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import median_abs_deviation

# Re-import the model classes from 26 to keep architectures in lock-step.
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from importlib import import_module
ckpt_mod = import_module("save_checkpoints")
Autoencoder = ckpt_mod.Autoencoder
AttentionMIL = ckpt_mod.AttentionMIL
build_demo_dict = ckpt_mod.build_demo_dict


COHORTS = ["Deng", "Good", "Harad", "Maurer"]


def load_adata(h5ad_path):
    """Same data prep as 24c: drop unlabeled, shift to [-1, 1]."""
    adata = sc.read_h5ad(h5ad_path)
    adata = adata[adata.obs["Response_3m"].notna()].copy()
    adata.obs["response_binary"] = adata.obs["Response_3m"].map({"OR": 1, "NR": 0}).astype(int)
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()
    adata.X = (adata.X - 0.5) * 2
    return adata


def latest_ckpt_dir(output_dir, variant_prefix="cohort_aware_antifit_ckpt_seed2"):
    """Pick the most recent timestamp-suffixed run dir."""
    candidates = sorted(glob.glob(os.path.join(output_dir, variant_prefix + "_*")))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint runs found under {output_dir}/{variant_prefix}_*"
        )
    return candidates[-1]


def load_manifest(result_dir):
    with open(os.path.join(result_dir, "checkpoint_manifest.pkl"), "rb") as f:
        return pickle.load(f)


def build_models(config, demo_dim, device):
    """Construct fresh AE + MIL with 24c hyperparameters; weights loaded later."""
    ae = Autoencoder(
        input_dim=154,
        latent_dim=config["latent_dim"],
        hidden_dim=config["ae_hidden_dim"],
        dropout=config["ae_dropout"],
    ).to(device)
    mil = AttentionMIL(
        input_dim=config["latent_dim"],
        num_classes=config["num_classes"],
        hidden_dim=config["hidden_dim"],
        dropout=config["mil_dropout"],
        sample_source_dim=None,  # use_sample_source=False in 24c
        use_gated_attention=config.get("use_gated_attention", True),
        demo_dim=demo_dim,
        film_hidden_dim=config.get("film_hidden_dim", 32),
    ).to(device)
    return ae, mil


def predict(mil_model, ae_model, cells_np, demo_vec, device):
    """Encode `cells_np` through frozen AE, then run frozen MIL with FiLM input.
    Returns scalar prob_OR (= class 1 probability).
    """
    ae_model.eval()
    mil_model.eval()
    with torch.no_grad():
        cells_t = torch.from_numpy(cells_np.astype(np.float32)).to(device)
        latent = ae_model.encode(cells_t)
        bag = [latent]
        demo_t = torch.from_numpy(np.asarray(demo_vec, dtype=np.float32)).to(device)
        # mil expects demo with batch dim or 1-D; we replicate the training
        # pattern (batch_size=1) by passing a [1, demo_dim] tensor.
        demo_t = demo_t.unsqueeze(0)
        logits = mil_model(bag, sample_source=None, demo=demo_t)
        probs = F.softmax(logits, dim=1)
        return float(probs[0, 1].item())


def perturb_one_tf(cells_np, tf_idx, direction, mad_value, mad_multiplier):
    """Per-cell additive (mad_multiplier × MAD) perturbation, clipped to [-1, 1]."""
    pert = cells_np.copy()
    shift = mad_multiplier * mad_value
    if direction == "up":
        pert[:, tf_idx] = np.clip(pert[:, tf_idx] + shift, -1.0, 1.0)
    elif direction == "down":
        pert[:, tf_idx] = np.clip(pert[:, tf_idx] - shift, -1.0, 1.0)
    else:
        raise ValueError(direction)
    return pert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_run_dir",
                    default=None,
                    help="Specific save_checkpoints output dir. "
                         "If omitted, use latest under default output_dir.")
    ap.add_argument("--ckpt_output_dir",
                    default="/home/users/kcytsui/tcellMIL/results/cohort_aware_antifit_ckpt/",
                    help="Parent directory that contains 26's run dir(s).")
    ap.add_argument("--h5ad_path",
                    default="/home/users/kcytsui/tcellMIL/data/cell_atlas_axicel_IP_scenic_tf_matrix_added_v2.h5ad")
    ap.add_argument("--metadata_csv_path",
                    default="/home/users/kcytsui/tcellMIL/data/Master_atlas_metadata.csv")
    ap.add_argument("--out_dir",
                    default="/home/users/kcytsui/tcellMIL/results/perturbation_24c/")
    ap.add_argument("--directions", default="up,down",
                    help="comma-separated subset of {up,down}")
    ap.add_argument("--mad_multiplier", type=float, default=3.0,
                    help="Per-cell additive shift size, in units of MAD. "
                         "Default 3.0 (matches original framework). "
                         "100.0 effectively saturates to clip bounds.")
    ap.add_argument("--n_test_folds", type=int, default=-1,
                    help="Limit to first N folds (smoke test). -1 = all.")
    ap.add_argument("--n_test_tfs", type=int, default=-1,
                    help="Limit to first N TFs (smoke test). -1 = all 154.")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mad_tag = f"{args.mad_multiplier:g}MAD"
    out_dir = os.path.join(args.out_dir, f"run_{timestamp}_{mad_tag}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1) Locate the checkpoint run + load manifest
    ckpt_run_dir = args.ckpt_run_dir or latest_ckpt_dir(args.ckpt_output_dir)
    print(f"Loading checkpoints from: {ckpt_run_dir}")
    manifest = load_manifest(ckpt_run_dir)
    config = manifest["config"]
    demo_dim = manifest["demo_dim"]
    folds = manifest["folds"]
    print(f"  Manifest seed={manifest['seed']}, "
          f"checkpoint-run accuracy={manifest['overall_accuracy']:.4f}")

    if args.n_test_folds > 0:
        folds = folds[: args.n_test_folds]
        print(f"  ** SMOKE: limiting to first {len(folds)} folds")

    # 2) Load adata + per-patient cell matrices
    print("Loading h5ad...")
    adata = load_adata(args.h5ad_path)
    tf_names = list(adata.var_names)
    n_tfs = len(tf_names)
    print(f"  {adata.n_obs} cells, {n_tfs} TFs")

    if args.n_test_tfs > 0:
        tf_indices_subset = list(range(min(args.n_test_tfs, n_tfs)))
        print(f"  ** SMOKE: limiting to first {len(tf_indices_subset)} TFs")
    else:
        tf_indices_subset = list(range(n_tfs))

    # 3) Per-patient cohort one-hot (FiLM input)
    cohort_patients = list(adata.obs["patient_id"].unique())
    demo_dict, _, demo_stats = build_demo_dict(args.metadata_csv_path, cohort_patients)
    print(f"  Cohort counts: {demo_stats['cohort_counts']}")

    # 4) Pre-load reference predictions for sanity check
    ref_pred_csv = os.path.join(ckpt_run_dir, "predictions.csv")
    ref_pred_df = pd.read_csv(ref_pred_csv).set_index("patient_id")
    print(f"  Reference predictions: {ref_pred_csv}")

    # 5) Build empty model shells (re-loaded per fold)
    ae_model, mil_model = build_models(config, demo_dim, device)

    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    print(f"Directions: {directions}")

    # ------------------------------------------------------------------
    # Loop: fold -> baseline pred -> per-TF perturbed pred (both directions)
    # ------------------------------------------------------------------
    rows = []
    sanity_rows = []
    t0 = time.time()
    for f_idx, fold in enumerate(folds):
        fold_t0 = time.time()
        test_patient = str(fold["test_patient"])
        true_label = int(fold["true_label"])
        ae_path = os.path.join(ckpt_run_dir, fold["ae_ckpt"])
        mil_path = os.path.join(ckpt_run_dir, fold["mil_ckpt"])

        # Patient cells & cohort one-hot
        mask = (adata.obs["patient_id"] == test_patient).values
        cells = adata.X[mask].astype(np.float32)
        n_cells = cells.shape[0]
        cohort_label = adata.obs.loc[mask, "Sample_source"].iloc[0] if mask.sum() > 0 else None
        demo_vec = np.array(demo_dict[test_patient], dtype=np.float32)

        # Load checkpoints
        ae_state = torch.load(ae_path, map_location=device, weights_only=True)
        mil_state = torch.load(mil_path, map_location=device, weights_only=True)
        ae_model.load_state_dict(ae_state)
        mil_model.load_state_dict(mil_state)

        # Baseline prediction (no perturbation)
        prob_base = predict(mil_model, ae_model, cells, demo_vec, device)

        # Sanity-check vs the predictions.csv from the ckpt run
        if test_patient in ref_pred_df.index:
            ref_prob = float(ref_pred_df.loc[test_patient, "prob_OR"])
            sanity_rows.append({
                "patient_id": test_patient,
                "true_label": true_label,
                "cohort": cohort_label,
                "ckpt_run_prob_OR": ref_prob,
                "this_run_prob_OR": prob_base,
                "abs_diff": abs(prob_base - ref_prob),
            })

        # Pre-compute MADs for all TFs across this patient's cells
        # median_abs_deviation default: scale='normal' multiplies by 1.4826.
        # The original framework used the SciPy default — we match it.
        mads = median_abs_deviation(cells, axis=0)  # shape (154,)

        # Per-TF perturbation
        for tf_idx in tf_indices_subset:
            tf_name = tf_names[tf_idx]
            for direction in directions:
                cells_pert = perturb_one_tf(cells, tf_idx, direction,
                                            mads[tf_idx], args.mad_multiplier)
                prob_pert = predict(mil_model, ae_model, cells_pert, demo_vec, device)
                rows.append({
                    "fold_idx": int(fold["fold_idx"]),
                    "patient_id": test_patient,
                    "true_label": true_label,
                    "cohort": cohort_label,
                    "n_cells": n_cells,
                    "tf": tf_name,
                    "tf_idx": int(tf_idx),
                    "direction": direction,
                    "mad": float(mads[tf_idx]),
                    "mad_multiplier": float(args.mad_multiplier),
                    "prob_OR_base": prob_base,
                    "prob_OR_perturbed": prob_pert,
                    "delta_prob_OR": prob_pert - prob_base,
                })

        fold_dt = time.time() - fold_t0
        print(f"  [{f_idx + 1:>2}/{len(folds)}] {test_patient} "
              f"(cohort={cohort_label}, n_cells={n_cells}) "
              f"baseline={prob_base:.4f} ref={ref_prob if test_patient in ref_pred_df.index else 'n/a'} "
              f"-- {fold_dt:.1f}s "
              f"({len(directions) * len(tf_indices_subset)} perturbations)")

        # Memory hygiene
        del ae_state, mil_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_dt = time.time() - t0
    print(f"\nFinished: {len(folds)} folds × {len(tf_indices_subset)} TFs × "
          f"{len(directions)} directions in {total_dt / 60:.1f} min")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    long_csv = os.path.join(out_dir, "perturbation_long.csv")
    df.to_csv(long_csv, index=False)
    print(f"Saved long-form CSV: {long_csv}  ({len(df)} rows)")

    sanity_df = pd.DataFrame(sanity_rows)
    sanity_csv = os.path.join(out_dir, "baseline_sanity_check.csv")
    sanity_df.to_csv(sanity_csv, index=False)
    print(f"Saved sanity check: {sanity_csv}")
    if len(sanity_df):
        worst = sanity_df["abs_diff"].max()
        mean = sanity_df["abs_diff"].mean()
        print(f"  Baseline reproducibility: max|Δ|={worst:.2e}, mean|Δ|={mean:.2e}")
        if worst > 1e-3:
            print("  WARNING: baseline predictions deviate from ckpt run by > 1e-3.")
            print("           The checkpoints may not be the ones that produced predictions.csv.")

    # Compact metadata file
    meta = {
        "ckpt_run_dir": ckpt_run_dir,
        "ckpt_run_accuracy": manifest["overall_accuracy"],
        "ckpt_run_auc": manifest["overall_auc"],
        "seed": manifest["seed"],
        "n_folds_processed": len(folds),
        "n_tfs_processed": len(tf_indices_subset),
        "directions": directions,
        "mad_multiplier": float(args.mad_multiplier),
        "perturbation_scheme":
            f"per_cell_additive_{args.mad_multiplier:g}MAD_clipped_-1_1",
        "h5ad_path": args.h5ad_path,
        "metadata_csv_path": args.metadata_csv_path,
        "tf_names": tf_names,
        "total_runtime_min": total_dt / 60,
    }
    with open(os.path.join(out_dir, "perturbation_metadata.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print("DONE")


if __name__ == "__main__":
    main()
