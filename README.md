# tcellMIL + cohort-aware FiLM

**tcellMIL** is a multiple-instance learning (MIL) framework that predicts durable (3-month) response to CAR T cell therapy from multi-cohort single-cell transcriptomic data, while retaining cell-state interpretability. It also enables in-silico perturbation of infusion-product cells to nominate genetic edits predicted to improve therapy outcome.

Explore the underlying data at **https://tcellwarehouse.com/**

## Installation

Requires **Python ≥ 3.10**.

```bash
git clone <repo-url>
cd tcellMIL_multimodal/scripts/tcellMIL_FiLM

# with uv (recommended)
uv pip install -r requirements.txt

# or with pip
pip install -r requirements.txt
```

This installs the runtime dependencies: `numpy`, `pandas`, `scipy`,
`scikit-learn`, `scanpy`, and `torch`. `wandb` (experiment tracking) is optional —
a no-op shim activates automatically if it is missing, so it is not required to
run the model.

## What's here

The code lives in [`scripts/tcellMIL_FiLM/`](scripts/tcellMIL_FiLM/):

| File | Role |
|---|---|
| `cohort_aware_film.{py,sh}` | The cohort-aware FiLM model — training + LOOCV. Reports the headline accuracy. |
| `save_checkpoints.{py,sh}` | Same model and pipeline, but exports per-fold checkpoints + a manifest (consumed by the perturbation engine). |
| `insilico_perturbation.{py,sh}` | In-silico TF over-expression / knockdown over the trained checkpoints. |
| `requirements.txt` | Runtime dependencies. |
| `README.md` | Full reproducibility doc — input schema, method, ASCII pipeline, and run steps. |

**See [`scripts/tcellMIL_FiLM/README.md`](scripts/tcellMIL_FiLM/README.md)** for
the authoritative, step-by-step reproducibility guide. The summary below is an
entry point.

## Model overview

```
per-cell expression  ──►  Autoencoder  ──►  Attention-MIL pooling  ──►  cohort-aware FiLM  ──►  classifier
 (SCENIC TF-regulon       (latent=48)      (bag embedding h, dim=64)         │                  (OR / NR)
  activity, AUCell)                                                          │
                          cohort one-hot (4-dim)  ──►  Linear(4 → 2·64)  ──►  (γ, β)
                                                                             │
                                                  h' = (1 + γ) ⊙ h + β  ◄────┘
```

A FiLM layer sits between the Attention-MIL bag-level pooling and the classifier.
For each patient, a one-hot of the source cohort is mapped by a single linear
layer to per-cohort modulation parameters `(γ, β)`; the bag embedding `h` is
residually modulated as `h' = (1 + γ) ⊙ h + β` before classification. The FiLM
linear layer is **zero-initialized**, so at step 0 the model reduces to the
no-FiLM model and learns cohort conditioning only if it helps. Hyperparameters
are tuned for the small-N regime (N = 64 patients).

## Input

The scripts expect a single-cell `h5ad` object whose `X` is a SCENIC
cell × regulon (AUCell) activity matrix — generate your own with
[pySCENIC](https://github.com/aertslab/pySCENIC). Expected metadata labels:

| `obs` column | Used for |
|---|---|
| `patient_id` | bag / patient grouping (the unit of LOOCV) |
| `Response_3m` | label — `"OR"` / `"NR"` (or numeric `1`/`0` in RNA-PCA mode) |
| `Sample_source` | cohort one-hot for FiLM |


## Usage

```bash
cd scripts/tcellMIL_FiLM

# 1. Train + LOOCV (headline metric)
TCMIL_H5AD=/path/to/scenic_auc.h5ad TCMIL_OUTDIR=/path/to/out/ \
  python cohort_aware_film.py

# 2. Export per-fold checkpoints (required before perturbation)
python save_checkpoints.py

# 3. In-silico TF perturbation over the trained checkpoints
python insilico_perturbation.py \
    --ckpt_output_dir /path/to/checkpoints/ \
    --out_dir /path/to/perturbation_out/ \
    --mad_multiplier 10   # perturbation size, in multiples of each regulon's MAD
```

The `.sh` launchers contain cluster-specific (SLURM) paths — adjust them, or run
the Python entry points directly as above.
