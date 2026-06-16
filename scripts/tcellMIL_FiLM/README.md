# tcellMIL + FiLM

Cohort-aware **F**eature-wise **L**inear **M**odulation extension of tcellMIL for
patient-level CAR-T response prediction from single-cell SCENIC TF-regulon
activity, plus an **in-silico TF-perturbation** engine that probes the trained
model.

This is a focused, code-only subset: the FiLM model, its checkpoint-export
companion, and the perturbation engine. 


## Method

```
per-cell SCENIC AUCell  ──►  Autoencoder  ──►  Attention-MIL pooling  ──►  cohort-aware FiLM  ──►  classifier
   (TF-regulon activity)     (latent_dim=48)     (bag embedding h, dim=64)        │                  (OR / NR)
                                                                                  │
                          cohort one-hot (4-dim)  ──►  Linear(4 → 2·64)  ──►  (γ, β)
                                                                                  │
                                                       h' = (1 + γ) ⊙ h + β  ◄────┘
```

A FiLM layer (Perez et al., AAAI 2018) sits between the Attention-MIL bag-level
pooling and the classifier head. For each patient, a 4-dim one-hot of the cohort
(`Sample_source`) is mapped by a single `Linear(4 → 2·hidden_dim)` to per-cohort
modulation parameters (γ, β); the bag embedding `h ∈ ℝ⁶⁴` is residually modulated
as `h' = (1 + γ) ⊙ h + β` before classification. The FiLM linear layer is
**zero-initialized**, so at step 0 `h' = h` and the model is identical to the
no-FiLM baseline. Anti-overfit hyperparameters tuned to N=64 patients:
`dropout=0.4`, `max_epochs=60`, `weight_decay=0.1`.

## Inputs

The scripts expect a **SCENIC AUCell h5ad** — one row per cell, TF-regulon
activity in `X` — whose `obs` contains:

| `obs` column | Used for |
|---|---|
| `patient_id` | bag / patient grouping (unit of LOOCV) |
| `Response_3m` | label — `"OR"`/`"NR"` (or numeric `1`/`0` for the RNA-PCA mode) |
| `Sample_source` | cohort one-hot for FiLM (`Deng`, `Good`, `Harad`, `Maurer`) |

A patient-metadata CSV is also referenced by the default SCENIC path. Supply your
own data and point the scripts at it via the env vars / CLI args below.

## Setup

```bash
uv pip install -r requirements.txt    # numpy, pandas, scipy, scikit-learn, scanpy, torch
```

`wandb` is optional — a no-op shim activates automatically if it is missing.

## Reproduce

All paths in the `.sh` launchers are Sherlock-specific; adjust them, or run the
Python entry points directly with the env vars / args shown here.

**1 — Train the model (headline metric).** Runs LOOCV over 3 seeds and writes
per-seed predictions + metrics:

```bash
python cohort_aware_film.py
# input mode + paths are overridable without editing the file:
TCMIL_INPUT_MODE=scenic \
TCMIL_H5AD=/path/to/scenic_auc.h5ad \
TCMIL_OUTDIR=/path/to/out/ \
  python cohort_aware_film.py
# TCMIL_INPUT_MODE=rna_pca runs the RNA-PCA-vs-SCENIC ablation (SCENIC defaults unchanged).
```

**2 — Export per-fold checkpoints** (required before perturbation). Same model
and pipeline as step 1, but saves each fold's autoencoder + MIL weights and a
`checkpoint_manifest.pkl`:

```bash
python save_checkpoints.py     # writes to <ckpt_output_dir>/cohort_aware_antifit_ckpt_seed2_<timestamp>/
```

**3 — In-silico TF perturbation.** Loads the checkpoints from step 2, verifies
its baseline predictions match the checkpoint run to float precision, then
perturbs (over-expresses / knocks down) MAD-selected TFs and records the change
in P(OR) per patient:

```bash
python insilico_perturbation.py \
    --ckpt_output_dir /path/to/checkpoints/ \
    --out_dir /path/to/perturbation_out/ \
    --mad_multiplier 10          # TF selection threshold (MAD multiples)
```

Output: long-form `perturbation_long.csv` (one row per patient × TF × direction)
plus `perturbation_metadata.pkl`.

## Files

| File | Role |
|---|---|
| `cohort_aware_film.{py,sh}` | Cohort-aware FiLM model — training + LOOCV; reports the headline accuracy. |
| `save_checkpoints.{py,sh}` | Same model/pipeline, exports per-fold checkpoints + manifest. Imported by the perturbation engine for its model definitions. |
| `insilico_perturbation.{py,sh}` | In-silico TF over-expression / knockdown engine over the trained checkpoints. |
| `requirements.txt` | Runtime dependencies. |
