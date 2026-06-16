#!/usr/bin/env python3
"""
save_checkpoints.py — same training pipeline as `cohort_aware_film.py`
(seed 2 only, the highest-accuracy seed: 0.7344 in the prior run), but with
ONE addition: per fold, persist the trained autoencoder + MIL model state
dicts to disk so a downstream perturbation script can load them and run
inference-only sweeps without retraining.

Output layout (under `result_dir`):

    predictions.csv
    loocv_results.pkl
    checkpoints/
        fold00_test_patient_<pid>_ae.pt
        fold00_test_patient_<pid>_mil.pt
        fold01_test_patient_<pid>_ae.pt
        ...

Each AE is the AE-state used for that fold's inference (after early-stopping
on the AE val split). Each MIL is the MIL-state at min train loss within the
60-epoch budget. Together they reproduce the exact prediction in
`predictions.csv` for the corresponding held-out patient.

Hyperparameters and seed RNG are kept identical to `cohort_aware_film.py` so
that re-running this script reproduces the prior `predictions_seed2.csv`
to within float precision; if it doesn't, the checkpoints are not the
ones that produced the 0.7344 numbers and the perturbation sweep is
moot.
"""

import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight

import wandb

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# MODEL DEFINITIONS  (verbatim from cohort_aware_film.py)
# ============================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class FiLMGenerator(nn.Module):
    def __init__(self, demo_dim, bag_dim, hidden_dim=32):
        super().__init__()
        self.bag_dim = bag_dim
        self.fc = nn.Linear(demo_dim, 2 * bag_dim)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, demo):
        out = self.fc(demo)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


class AttentionMIL(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dim=128,
                 dropout=0.0, sample_source_dim=None, use_gated_attention=False,
                 demo_dim=None, film_hidden_dim=32):
        super().__init__()
        self.use_sample_source = sample_source_dim is not None
        self.use_gated_attention = use_gated_attention
        self.use_film = demo_dim is not None and demo_dim > 0

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if use_gated_attention:
            self.attention_V = nn.Linear(hidden_dim, hidden_dim)
            self.attention_U = nn.Linear(hidden_dim, hidden_dim)
            self.attention_w = nn.Linear(hidden_dim, 1)
        else:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        if self.use_film:
            self.film = FiLMGenerator(demo_dim=demo_dim, bag_dim=hidden_dim,
                                      hidden_dim=film_hidden_dim)

        if self.use_sample_source:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim + sample_source_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x, sample_source=None, demo=None, return_attention=False,
                return_film=False):
        device = next(self.parameters()).device
        batch_size = len(x)
        all_logits = []
        all_attention_weights = []
        all_film_params = []

        for i in range(batch_size):
            instances = x[i]
            instance_features = self.feature_extractor(instances)
            if self.use_gated_attention:
                attention_scores = self.attention_w(
                    torch.tanh(self.attention_V(instance_features)) *
                    torch.sigmoid(self.attention_U(instance_features))
                )
            else:
                attention_scores = self.attention(instance_features)
            attention_weights = F.softmax(attention_scores, dim=0)
            weighted_features = torch.sum(instance_features * attention_weights, dim=0)

            if self.use_film and demo is not None:
                demo_i = demo[i] if demo.dim() > 1 else demo.squeeze(0)
                gamma, beta = self.film(demo_i)
                weighted_features = (1.0 + gamma) * weighted_features + beta
                if return_film:
                    all_film_params.append((gamma.detach().cpu().numpy(),
                                            beta.detach().cpu().numpy()))

            if self.use_sample_source and sample_source is not None:
                sample_source_i = sample_source[i] if sample_source.dim() > 1 else sample_source.squeeze(0)
                combined = torch.cat([weighted_features, sample_source_i], dim=0)
                logits = self.classifier(combined)
            else:
                logits = self.classifier(weighted_features)

            all_logits.append(logits)
            all_attention_weights.append(attention_weights)

        logits = torch.stack(all_logits)
        outputs = [logits]
        if return_attention:
            outputs.append(all_attention_weights)
        if return_film:
            outputs.append(all_film_params)
        if len(outputs) == 1:
            return logits
        return tuple(outputs)


# ============================================================================
# DEMOGRAPHIC METADATA HELPERS
# ============================================================================

def build_demo_dict(metadata_csv_path, patient_ids):
    df = pd.read_csv(metadata_csv_path)
    df = df.set_index("patient_id")
    df = df.loc[df.index.intersection(patient_ids)].copy()

    cohorts = ["Deng", "Good", "Harad", "Maurer"]

    demo_dict = {}
    for pid in patient_ids:
        cohort = df.loc[pid, "Sample_source"] if pid in df.index else None
        one_hot = [1.0 if cohort == c else 0.0 for c in cohorts]
        demo_dict[pid] = one_hot

    stats = {
        "cohorts": cohorts,
        "n_patients": len(demo_dict),
        "cohort_counts": {c: int((df["Sample_source"] == c).sum()) for c in cohorts},
    }
    return demo_dict, 4, stats


# ============================================================================
# DATASET
# ============================================================================

class ResponsePatientBagDataset(Dataset):
    def __init__(self, adata, patient_col="patient_id", label_col="response_binary",
                 sample_sources_order=None, demo_dict=None):
        self.adata = adata
        self.patient_col = patient_col
        self.label_col = label_col
        self.demo_dict = demo_dict or {}

        self.patients = list(adata.obs[patient_col].unique())

        self.patient_metadata = {}
        if "Sample_source" in adata.obs.columns and sample_sources_order is not None:
            patient_source_map = (
                adata.obs.groupby(patient_col, observed=True)["Sample_source"]
                .first().to_dict()
            )
            for patient in self.patients:
                source = patient_source_map.get(patient, None)
                one_hot = [1.0 if s == source else 0.0 for s in sample_sources_order]
                self.patient_metadata[patient] = one_hot

        self.patient_bags = {}
        self.patient_labels = {}
        for patient in self.patients:
            mask = adata.obs[patient_col] == patient
            indices = np.where(mask)[0]
            data = adata.X[indices]
            if not isinstance(data, np.ndarray):
                data = data.toarray()
            self.patient_bags[patient] = data

            label_vals = adata.obs.loc[mask, label_col].values
            self.patient_labels[patient] = int(label_vals[0])

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        bag = torch.FloatTensor(self.patient_bags[patient])
        label = torch.tensor(self.patient_labels[patient], dtype=torch.long)

        if self.patient_metadata:
            one_hot = torch.tensor(self.patient_metadata[patient], dtype=torch.float)
        else:
            one_hot = torch.zeros(1)

        if self.demo_dict and patient in self.demo_dict:
            demo = torch.tensor(self.demo_dict[patient], dtype=torch.float)
        else:
            demo = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float)

        return bag, label, patient, one_hot, demo


# ============================================================================
# HELPERS
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_prepare_data(config):
    print("Loading h5ad...")
    adata = sc.read_h5ad(config["h5ad_path"])
    print(f"  Loaded: {adata.n_obs} cells, {adata.n_vars} TFs, "
          f"{adata.obs['patient_id'].nunique()} patients")

    if "Response_3m" not in adata.obs.columns:
        raise ValueError("Response_3m column not found in adata.obs")

    adata = adata[adata.obs["Response_3m"].notna()].copy()
    adata.obs["response_binary"] = adata.obs["Response_3m"].map({"OR": 1, "NR": 0})

    unmapped = adata.obs["response_binary"].isna().sum()
    if unmapped > 0:
        adata = adata[adata.obs["response_binary"].notna()].copy()

    adata.obs["response_binary"] = adata.obs["response_binary"].astype(int)

    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()
    adata.X = (adata.X - 0.5) * 2

    return adata


def train_autoencoder_fold(train_cells, val_cells, config, device):
    input_dim = train_cells.shape[1]
    latent_dim = config["latent_dim"]
    ae_hidden = config.get("ae_hidden_dim", 64)
    model = Autoencoder(input_dim, latent_dim, hidden_dim=ae_hidden,
                        dropout=config["ae_dropout"]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["ae_learning_rate"],
        weight_decay=config["ae_weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    train_tensor = torch.FloatTensor(train_cells)
    val_tensor = torch.FloatTensor(val_cells)
    train_loader = DataLoader(
        TensorDataset(train_tensor, train_tensor),
        batch_size=config["ae_batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor, val_tensor),
        batch_size=config["ae_batch_size"],
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(config["ae_max_epochs"]):
        model.train()
        for data, _ in train_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["ae_patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model


def encode_cells(autoencoder, cells, device, batch_size=1024):
    autoencoder.eval()
    latent_parts = []
    n = cells.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = torch.FloatTensor(cells[start:start + batch_size]).to(device)
            z = autoencoder.encode(batch).cpu().numpy()
            latent_parts.append(z)
    return np.vstack(latent_parts)


# ============================================================================
# LOOCV with checkpoint saving
# ============================================================================

def run_loocv(config, adata, seed, demo_dict, demo_dim,
              variant_name="cohort_aware_antifit_ckpt"):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(config["output_dir"],
                              f"{variant_name}_seed{seed}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    run_name = f"{variant_name}_seed{seed}_{timestamp}"
    wandb.init(
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        tags=config["wandb_tags"],
        name=run_name,
        config={**config, "seed": seed},
        settings=wandb.Settings(init_timeout=300),
    )

    sample_sources_order = sorted(adata.obs["Sample_source"].unique())
    sample_source_dim = len(sample_sources_order) if config["use_sample_source"] else None

    patient_info = (
        adata.obs.groupby("patient_id", observed=True)["response_binary"]
        .first().reset_index()
    )
    patients = patient_info["patient_id"].values
    labels = patient_info["response_binary"].values
    n_patients = len(patients)

    print(f"Starting LOOCV for {n_patients} patients (seed={seed})...")

    all_true, all_pred, all_prob, all_pids = [], [], [], []
    cv_results = {
        "fold_metrics": [],
        "patient_predictions": {},
        "attention_weights": {},
        "film_params": {},
    }

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=labels,
    )
    minority_class = 0 if np.sum(labels == 0) < np.sum(labels == 1) else 1
    class_weights[minority_class] *= config.get("minority_boost", 1.5)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights (boosted minority={minority_class}): {class_weights}")

    fold_index_records = []  # for ckpt manifest

    for fold_idx, test_patient in enumerate(patients):
        test_label = labels[fold_idx]
        train_patients = [p for p in patients if p != test_patient]

        print(f"\n  Fold {fold_idx + 1}/{n_patients}: test patient={test_patient} "
              f"(true label={test_label})")

        train_mask = adata.obs["patient_id"].isin(train_patients)
        train_cells = adata.X[train_mask.values]
        if not isinstance(train_cells, np.ndarray):
            train_cells = train_cells.toarray()

        n_train = train_cells.shape[0]
        rng = np.random.RandomState(seed + fold_idx)
        val_indices = rng.choice(n_train, size=max(1, int(0.1 * n_train)), replace=False)
        train_indices = np.setdiff1d(np.arange(n_train), val_indices)

        ae_train = train_cells[train_indices]
        ae_val = train_cells[val_indices]

        autoencoder = train_autoencoder_fold(ae_train, ae_val, config, device)

        # Persist AE state BEFORE moving on to MIL training (no shape mutation)
        ae_ckpt_path = os.path.join(
            ckpt_dir, f"fold{fold_idx:02d}_test_patient_{test_patient}_ae.pt"
        )
        torch.save(
            {k: v.detach().cpu() for k, v in autoencoder.state_dict().items()},
            ae_ckpt_path,
        )

        all_cells = adata.X
        if not isinstance(all_cells, np.ndarray):
            all_cells = all_cells.toarray()
        latent_all = encode_cells(autoencoder, all_cells, device)

        adata_latent = sc.AnnData(latent_all)
        adata_latent.obs = adata.obs.copy()
        adata_latent.obs.index = adata.obs.index.copy()

        train_adata = adata_latent[adata_latent.obs["patient_id"].isin(train_patients)].copy()
        test_adata = adata_latent[adata_latent.obs["patient_id"] == test_patient].copy()

        train_dataset = ResponsePatientBagDataset(
            train_adata,
            sample_sources_order=sample_sources_order if config["use_sample_source"] else None,
            demo_dict=demo_dict,
        )
        test_dataset = ResponsePatientBagDataset(
            test_adata,
            sample_sources_order=sample_sources_order if config["use_sample_source"] else None,
            demo_dict=demo_dict,
        )

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        mil_model = AttentionMIL(
            input_dim=config["latent_dim"],
            num_classes=config["num_classes"],
            hidden_dim=config["hidden_dim"],
            dropout=config["mil_dropout"],
            sample_source_dim=sample_source_dim,
            use_gated_attention=config.get("use_gated_attention", False),
            demo_dim=demo_dim,
            film_hidden_dim=config.get("film_hidden_dim", 32),
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(
            mil_model.parameters(),
            lr=config["mil_learning_rate"],
            weight_decay=config["mil_weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5, factor=0.5
        )

        best_train_loss = float("inf")
        best_mil_state = None
        epochs_no_improve = 0

        for epoch in range(config["mil_max_epochs"]):
            mil_model.train()
            train_loss = 0.0
            for bags, batch_labels, _, one_hot_ss, demo_batch in train_loader:
                bags = [bag.to(device) for bag in bags]
                batch_labels = batch_labels.to(device)
                one_hot_ss = one_hot_ss.to(device) if config["use_sample_source"] else None
                demo_batch = demo_batch.to(device)

                logits = mil_model(bags, sample_source=one_hot_ss, demo=demo_batch)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            scheduler.step(train_loss)

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_mil_state = {k: v.cpu().clone() for k, v in mil_model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config["mil_patience"]:
                    break

        if best_mil_state is not None:
            mil_model.load_state_dict(best_mil_state)
        mil_model.to(device)

        mil_ckpt_path = os.path.join(
            ckpt_dir, f"fold{fold_idx:02d}_test_patient_{test_patient}_mil.pt"
        )
        torch.save(
            {k: v.detach().cpu() for k, v in mil_model.state_dict().items()},
            mil_ckpt_path,
        )

        fold_index_records.append({
            "fold_idx": int(fold_idx),
            "test_patient": str(test_patient),
            "true_label": int(test_label),
            "ae_ckpt": os.path.relpath(ae_ckpt_path, result_dir),
            "mil_ckpt": os.path.relpath(mil_ckpt_path, result_dir),
        })

        mil_model.eval()
        with torch.no_grad():
            for bags, batch_labels, patient_ids, one_hot_ss, demo_batch in test_loader:
                bags = [bag.to(device) for bag in bags]
                batch_labels = batch_labels.to(device)
                one_hot_ss = one_hot_ss.to(device) if config["use_sample_source"] else None
                demo_batch = demo_batch.to(device)

                logits, attn_weights, film_params = mil_model(
                    bags, sample_source=one_hot_ss, demo=demo_batch,
                    return_attention=True, return_film=True,
                )
                probs = F.softmax(logits, dim=1)
                _, pred = torch.max(logits, 1)

                pred_np = pred.cpu().numpy()[0]
                true_np = batch_labels.cpu().numpy()[0]
                prob_np = probs.cpu().numpy()[0]
                patient_id = patient_ids[0]
                pos_prob = float(prob_np[1])

                all_true.append(true_np)
                all_pred.append(pred_np)
                all_prob.append(pos_prob)
                all_pids.append(patient_id)

                correct = bool(pred_np == true_np)

                cv_results["patient_predictions"][patient_id] = {
                    "true_label": int(true_np),
                    "predicted_label": int(pred_np),
                    "probabilities": prob_np.tolist(),
                    "correct": correct,
                }
                cv_results["attention_weights"][patient_id] = [
                    w.cpu().numpy() for w in attn_weights
                ]
                cv_results["film_params"][patient_id] = film_params

                fold_metrics = {
                    "patient_id": patient_id,
                    "fold": fold_idx,
                    "accuracy": 1.0 if correct else 0.0,
                    "true_label": int(true_np),
                    "predicted_label": int(pred_np),
                    "prob_positive": pos_prob,
                }
                cv_results["fold_metrics"].append(fold_metrics)
                print(f"    -> Prediction: {pred_np} (true: {true_np}) "
                      f"prob_OR={pos_prob:.3f} {'CORRECT' if correct else 'WRONG'}")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)

    overall_accuracy = accuracy_score(all_true, all_pred)
    overall_balanced_acc = balanced_accuracy_score(all_true, all_pred)
    overall_precision = precision_score(all_true, all_pred, zero_division=0)
    overall_recall = recall_score(all_true, all_pred, zero_division=0)
    overall_f1 = f1_score(all_true, all_pred, zero_division=0)
    try:
        overall_auc = roc_auc_score(all_true, all_prob)
    except ValueError:
        overall_auc = float("nan")
    conf_matrix = confusion_matrix(all_true, all_pred)

    cv_results["overall_metrics"] = {
        "accuracy": float(overall_accuracy),
        "balanced_accuracy": float(overall_balanced_acc),
        "precision": float(overall_precision),
        "recall": float(overall_recall),
        "f1": float(overall_f1),
        "auc": float(overall_auc),
        "confusion_matrix": conf_matrix.tolist(),
    }

    print(f"\n===== LOOCV Results (seed={seed}) =====")
    print(f"Accuracy:          {overall_accuracy:.4f} "
          f"({int(np.sum(all_pred == all_true))}/{len(all_true)} correct)")
    print(f"AUC:               {overall_auc:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):\n  {conf_matrix}")

    pred_df = pd.DataFrame({
        "patient_id": all_pids,
        "true_label": all_true,
        "predicted_label": all_pred,
        "prob_OR": all_prob,
        "correct": all_pred == all_true,
    })
    pred_df.to_csv(os.path.join(result_dir, "predictions.csv"), index=False)

    with open(os.path.join(result_dir, "loocv_results.pkl"), "wb") as f:
        pickle.dump(cv_results, f)

    # Manifest so insilico_perturbation.py can find checkpoints by fold
    manifest = {
        "seed": int(seed),
        "result_dir": result_dir,
        "ckpt_dir": ckpt_dir,
        "config": config,
        "demo_dim": int(demo_dim),
        "cohorts": ["Deng", "Good", "Harad", "Maurer"],
        "sample_sources_order": list(sample_sources_order),
        "use_sample_source": bool(config["use_sample_source"]),
        "folds": fold_index_records,
        "overall_accuracy": float(overall_accuracy),
        "overall_auc": float(overall_auc),
    }
    with open(os.path.join(result_dir, "checkpoint_manifest.pkl"), "wb") as f:
        pickle.dump(manifest, f)
    print(f"\nCheckpoints + manifest saved under: {ckpt_dir}")

    wandb.log({
        "overall/accuracy": overall_accuracy,
        "overall/balanced_accuracy": overall_balanced_acc,
        "overall/auc": overall_auc,
        "overall/seed": seed,
    })
    wandb.finish()
    return cv_results, result_dir


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    SEEDS = [2]  # ONLY seed 2 — highest-acc seed (0.7344) per task briefing.

    CONFIG = {
        "h5ad_path": "/home/users/kcytsui/tcellMIL/data/cell_atlas_axicel_IP_scenic_tf_matrix_added_v2.h5ad",
        "metadata_csv_path": "/home/users/kcytsui/tcellMIL/data/Master_atlas_metadata.csv",
        "output_dir": "/home/users/kcytsui/tcellMIL/results/cohort_aware_antifit_ckpt/",

        # Autoencoder
        "latent_dim": 48,
        "ae_hidden_dim": 96,
        "ae_learning_rate": 1e-3,
        "ae_weight_decay": 1e-4,
        "ae_max_epochs": 200,
        "ae_patience": 15,
        "ae_batch_size": 256,
        "ae_dropout": 0.1,

        # MIL — 24c anti-overfit knobs
        "hidden_dim": 64,
        "mil_learning_rate": 1e-3,
        "mil_weight_decay": 1e-1,
        "mil_max_epochs": 60,
        "mil_patience": 15,
        "mil_dropout": 0.4,
        "use_gated_attention": True,
        "num_classes": 2,

        "film_hidden_dim": 32,
        "minority_boost": 1.5,
        "use_sample_source": False,

        "wandb_project": "cohort_aware_antifit_ckpt",
        "wandb_entity": "mackall_lab",
        "wandb_tags": ["response_prediction", "perturbation_setup", "ckpt_save",
                       "film", "seed2"],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    adata = load_and_prepare_data(CONFIG)

    cohort_patients = list(adata.obs["patient_id"].unique())
    demo_dict, demo_dim, demo_stats = build_demo_dict(
        CONFIG["metadata_csv_path"], cohort_patients
    )
    print(f"\nFiLM encoding (dim={demo_dim}): cohort one-hot {demo_stats['cohorts']}")
    print(f"  Patients: {demo_stats['n_patients']}")
    print(f"  Cohort counts: {demo_stats['cohort_counts']}")

    seed_metrics = []
    for seed in SEEDS:
        print("\n" + "#" * 80)
        print(f"# SEED {seed}")
        print("#" * 80)
        results, _ = run_loocv(CONFIG, adata, seed, demo_dict, demo_dim,
                               variant_name="cohort_aware_antifit_ckpt")
        m = results["overall_metrics"]
        seed_metrics.append({"seed": seed, **{k: m[k] for k in
            ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"]}})

    metrics_df = pd.DataFrame(seed_metrics)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(
        os.path.join(CONFIG["output_dir"], "ckpt_run_summary.csv"), index=False
    )
    print("\nDONE — checkpoints ready for insilico_perturbation.py")
