#!/usr/bin/env python3
"""
FiLM (Feature-wise Linear Modulation) conditioning on age + sex at the
bag-level head.

For each patient, we build a small demographic vector
    [age_z, gender_bin, missing_mask]
A small FiLMGenerator MLP maps the demo vector to (gamma, beta) of the
bag embedding's dimension. After attention pooling produces the bag
embedding `h` (dim = hidden_dim), we apply a residual FiLM transform:

    h' = (1 + gamma) * h + beta

The final layer of FiLMGenerator is initialized to all zeros so gamma=0
and beta=0 at start, which makes h' == h at init -- model is identical
to the no-FiLM baseline at step 0, gradient still flows through the
generator (because of the residual `1 +` term).

Reference: Perez et al., "FiLM: Visual Reasoning with a General
Conditioning Layer", AAAI 2018.

Based on: 05_multiseed_response_gated.py
Seeds: [0, 1, 2]
"""

import os
import sys
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

try:
    import wandb
    if not hasattr(wandb, "init"):
        raise ImportError("wandb missing/shadowed")
except Exception:
    # No-op wandb shim (env may lack wandb, or a local ./wandb dir shadows it).
    import types as _types, sys as _sys
    class _WandbNoop:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return self
        def __getattr__(self, _n): return self
    wandb = _types.ModuleType("wandb")
    _noop = _WandbNoop()
    for _fn in ("init", "log", "finish", "watch", "save", "Table", "Image"):
        setattr(wandb, _fn, _noop)
    wandb.config = _noop
    _sys.modules["wandb"] = wandb

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class Autoencoder(nn.Module):
    """Autoencoder for SCENIC AUC matrix: input_dim -> hidden -> latent -> hidden -> input_dim."""

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
    """
    Linear-only FiLM: Linear(demo_dim -> 2*bag_dim). Zero-init.
    """

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
    """Attention-based MIL with optional gated attention and optional
    FiLM conditioning of the bag embedding on a demographic vector."""

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
        """
        Parameters
        ----------
        x : list of tensors, each [num_instances, features] (one per bag in batch)
        sample_source : tensor [batch_size, sample_source_dim] or None
        demo : tensor [batch_size, demo_dim] or None
        return_attention : bool
        return_film : bool -- if True, also return list of (gamma, beta)
            tensors (one per bag) for inspection.
        """
        device = next(self.parameters()).device
        batch_size = len(x)
        all_logits = []
        all_attention_weights = []
        all_film_params = []

        for i in range(batch_size):
            instances = x[i]  # [num_instances, features]
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

            # FiLM conditioning on demographics (residual form)
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
    """
    24a: Cohort-aware FiLM. Demo vector = one-hot Sample_source (4-dim:
    Deng/Good/Harad/Maurer). Lets FiLM learn cohort-specific bag-level
    adjustments — addresses the diagnostic finding that demographics
    have a cohort-confounded relationship with response (Harad has a
    strong age+sex effect, others don't).
    """
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


def build_demo_dict_from_adata(adata, patient_ids, patient_col="patient_id"):
    """Cohort one-hot from the h5ad's own Sample_source. Used when patient IDs
    don't match the external metadata CSV (e.g. the RNA-PCA matrix uses CAR-IDs)."""
    cohorts = ["Deng", "Good", "Harad", "Maurer"]
    pmap = (
        adata.obs.groupby(patient_col, observed=True)["Sample_source"]
        .first().to_dict()
    )
    demo_dict = {
        pid: [1.0 if pmap.get(pid) == c else 0.0 for c in cohorts]
        for pid in patient_ids
    }
    stats = {
        "cohorts": cohorts,
        "n_patients": len(demo_dict),
        "cohort_counts": {c: int((adata.obs["Sample_source"] == c).sum()) for c in cohorts},
    }
    return demo_dict, 4, stats


# ============================================================================
# DATASET CLASS for Response prediction with demographics
# ============================================================================

class ResponsePatientBagDataset(Dataset):
    """MIL dataset: each bag = cells from one patient, label = Response_3m (OR=1, NR=0).

    Now also returns a demographic tensor (age_z, gender_bin, missing_mask)
    per patient for FiLM conditioning.
    """

    def __init__(self, adata, patient_col="patient_id", label_col="response_binary",
                 sample_sources_order=None, demo_dict=None):
        self.adata = adata
        self.patient_col = patient_col
        self.label_col = label_col
        self.demo_dict = demo_dict or {}

        self.patients = list(adata.obs[patient_col].unique())

        # Build one-hot for Sample_source
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

        # Build bags and labels
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
            # 24a: 4-dim cohort one-hot fallback (all zeros = unknown cohort)
            demo = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float)

        return bag, label, patient, one_hot, demo


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_prepare_data(config):
    """Load h5ad, filter to patients with Response_3m labels, create binary labels."""
    print("Loading h5ad...")
    adata = sc.read_h5ad(config["h5ad_path"])
    print(f"  Loaded: {adata.n_obs} cells, {adata.n_vars} TFs, "
          f"{adata.obs['patient_id'].nunique()} patients")

    if "Response_3m" not in adata.obs.columns:
        raise ValueError("Response_3m column not found in adata.obs")

    print(f"  Response_3m values: {adata.obs['Response_3m'].value_counts().to_dict()}")

    adata = adata[adata.obs["Response_3m"].notna()].copy()
    _resp = adata.obs["Response_3m"]
    if pd.api.types.is_numeric_dtype(_resp):
        # RNA-PCA h5ad stores Response_3m as int (OR=1, NR=0)
        adata.obs["response_binary"] = _resp.astype(int)
    else:
        adata.obs["response_binary"] = _resp.map({"OR": 1, "NR": 0})

    unmapped = adata.obs["response_binary"].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped} cells with unmapped Response_3m values, dropping them")
        adata = adata[adata.obs["response_binary"].notna()].copy()

    adata.obs["response_binary"] = adata.obs["response_binary"].astype(int)

    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()
    input_mode = config.get("input_mode", "scenic")
    if input_mode == "rna_pca":
        # Continuous PCA features: per-feature min-max to [-1, 1] so the
        # Tanh-decoder AE stays valid (parallels the SCENIC [0,1]->[-1,1] map).
        xmin = adata.X.min(axis=0, keepdims=True)
        xmax = adata.X.max(axis=0, keepdims=True)
        adata.X = 2.0 * (adata.X - xmin) / np.maximum(xmax - xmin, 1e-8) - 1.0
    else:
        adata.X = (adata.X - 0.5) * 2

    n_patients = adata.obs["patient_id"].nunique()
    patient_labels = adata.obs.groupby("patient_id")["response_binary"].first()
    n_or = int(patient_labels.sum())
    n_nr = n_patients - n_or
    print(f"\n  Final dataset: {adata.n_obs} cells, {n_patients} patients")
    print(f"  OR (responders): {n_or} patients")
    print(f"  NR (non-responders): {n_nr} patients")
    print(f"  Sample sources: {sorted(adata.obs['Sample_source'].unique())}")

    return adata


def train_autoencoder_fold(train_cells, val_cells, config, device):
    input_dim = train_cells.shape[1]
    latent_dim = config["latent_dim"]
    ae_hidden = config.get("ae_hidden_dim", 64)
    model = Autoencoder(input_dim, latent_dim, hidden_dim=ae_hidden, dropout=config["ae_dropout"]).to(device)
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
        train_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

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
# SINGLE-SEED LOOCV PIPELINE
# ============================================================================

def run_loocv(config, adata, seed, demo_dict, demo_dim, variant_name="cohort_aware_antifit"):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(config["output_dir"], f"{variant_name}_seed{seed}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    run_name = f"{variant_name}_seed{seed}_{timestamp}"
    run = wandb.init(
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        tags=config["wandb_tags"],
        name=run_name,
        config={**config, "seed": seed},
    )

    sample_sources_order = sorted(adata.obs["Sample_source"].unique())
    sample_source_dim = len(sample_sources_order) if config["use_sample_source"] else None

    wandb.config.update({
        "variant": variant_name,
        "seed": seed,
        "n_cells": adata.n_obs,
        "n_tfs": adata.n_vars,
        "n_patients": adata.obs["patient_id"].nunique(),
        "sample_sources": sample_sources_order,
        "sample_source_dim": sample_source_dim,
        "demo_dim": demo_dim,
    }, allow_val_change=True)

    patient_info = (
        adata.obs.groupby("patient_id", observed=True)["response_binary"]
        .first().reset_index()
    )
    patients = patient_info["patient_id"].values
    labels = patient_info["response_binary"].values
    n_patients = len(patients)

    print(f"Starting LOOCV for {n_patients} patients (seed={seed})...")

    all_true = []
    all_pred = []
    all_prob = []
    all_pids = []
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

    for fold_idx, test_patient in enumerate(patients):
        test_label = labels[fold_idx]
        train_patients = [p for p in patients if p != test_patient]

        print(f"\n  Fold {fold_idx + 1}/{n_patients}: test patient={test_patient} "
              f"(true label={test_label})")

        train_mask = adata.obs["patient_id"].isin(train_patients)
        test_mask = adata.obs["patient_id"] == test_patient

        train_cells = adata.X[train_mask.values]
        test_cells = adata.X[test_mask.values]

        if not isinstance(train_cells, np.ndarray):
            train_cells = train_cells.toarray()
        if not isinstance(test_cells, np.ndarray):
            test_cells = test_cells.toarray()

        n_train = train_cells.shape[0]
        rng = np.random.RandomState(seed + fold_idx)
        val_indices = rng.choice(n_train, size=max(1, int(0.1 * n_train)), replace=False)
        train_indices = np.setdiff1d(np.arange(n_train), val_indices)

        ae_train = train_cells[train_indices]
        ae_val = train_cells[val_indices]

        autoencoder = train_autoencoder_fold(ae_train, ae_val, config, device)

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
            train_correct = 0
            train_total = 0

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
                _, preds = torch.max(logits, 1)
                train_total += batch_labels.size(0)
                train_correct += (preds == batch_labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0

            scheduler.step(train_loss)

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}/{config['mil_max_epochs']}: "
                      f"loss={train_loss:.4f}, acc={train_acc:.4f}")

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

                wandb.log({
                    "fold": fold_idx,
                    f"fold/{patient_id}/true_label": int(true_np),
                    f"fold/{patient_id}/predicted_label": int(pred_np),
                    f"fold/{patient_id}/prob_OR": pos_prob,
                    f"fold/{patient_id}/correct": int(correct),
                })

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

    class_metrics = {}
    for c in range(config["num_classes"]):
        tp = np.sum((all_true == c) & (all_pred == c))
        actual = np.sum(all_true == c)
        predicted = np.sum(all_pred == c)
        class_metrics[f"class_{c}"] = {
            "precision": tp / predicted if predicted > 0 else 0,
            "recall": tp / actual if actual > 0 else 0,
            "count": int(actual),
        }

    cv_results["overall_metrics"] = {
        "accuracy": float(overall_accuracy),
        "balanced_accuracy": float(overall_balanced_acc),
        "precision": float(overall_precision),
        "recall": float(overall_recall),
        "f1": float(overall_f1),
        "auc": float(overall_auc),
        "confusion_matrix": conf_matrix.tolist(),
        "class_metrics": class_metrics,
    }

    print(f"\n===== LOOCV Results (seed={seed}) =====")
    print(f"Accuracy:          {overall_accuracy:.4f} "
          f"({int(np.sum(all_pred == all_true))}/{len(all_true)} correct)")
    print(f"Balanced Accuracy: {overall_balanced_acc:.4f}")
    print(f"Precision:         {overall_precision:.4f}")
    print(f"Recall:            {overall_recall:.4f}")
    print(f"F1 Score:          {overall_f1:.4f}")
    print(f"AUC:               {overall_auc:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"  NR(0)  OR(1)")
    print(f"  {conf_matrix}")

    wandb.log({
        "overall/accuracy": overall_accuracy,
        "overall/balanced_accuracy": overall_balanced_acc,
        "overall/precision": overall_precision,
        "overall/recall": overall_recall,
        "overall/f1": overall_f1,
        "overall/auc": overall_auc,
        "overall/seed": seed,
    })

    cm_table = wandb.Table(
        columns=["True Label", "Predicted Label", "Count"],
        data=[
            ["NR", "NR", int(conf_matrix[0, 0])],
            ["NR", "OR", int(conf_matrix[0, 1])],
            ["OR", "NR", int(conf_matrix[1, 0])],
            ["OR", "OR", int(conf_matrix[1, 1])],
        ],
    )
    wandb.log({"confusion_matrix": cm_table})

    patient_table = wandb.Table(
        columns=["patient_id", "true_label", "predicted_label", "prob_OR", "correct"]
    )
    for pid, info in cv_results["patient_predictions"].items():
        patient_table.add_data(
            pid,
            info["true_label"],
            info["predicted_label"],
            info["probabilities"][1],
            info["correct"],
        )
    wandb.log({"patient_results": patient_table})

    pred_df = pd.DataFrame({
        "patient_id": all_pids,
        "true_label": all_true,
        "predicted_label": all_pred,
        "prob_OR": all_prob,
        "correct": all_pred == all_true,
    })
    pred_csv_path = os.path.join(result_dir, "predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"  Predictions saved to: {pred_csv_path}")

    pkl_path = os.path.join(result_dir, "loocv_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(cv_results, f)
    print(f"  Full results pickle saved to: {pkl_path}")

    wandb.finish()

    return cv_results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    SEEDS = [0, 1, 2]

    # Input mode: "scenic" (default, production) or "rna_pca" (RNA-vs-SCENIC ablation).
    # Override via env vars so the SCENIC defaults stay byte-identical.
    _INPUT_MODE = os.environ.get("TCMIL_INPUT_MODE", "scenic")
    _DEFAULT_H5AD = {
        "scenic": "/home/users/kcytsui/tcellMIL/data/cell_atlas_axicel_IP_scenic_tf_matrix_added_v2.h5ad",
        "rna_pca": "/home/users/kcytsui/tcellMIL/data/cell_atlas_axicel_IP_RNA_assay_PCA.h5ad",
    }[_INPUT_MODE]
    _DEFAULT_OUT = {
        "scenic": "/home/users/kcytsui/tcellMIL/results/cohort_aware_antifit/",
        "rna_pca": "/home/users/kcytsui/tcellMIL/results/cohort_aware_antifit_rna_pca/",
    }[_INPUT_MODE]

    CONFIG = {
        # Paths (Sherlock)
        "input_mode": _INPUT_MODE,
        "h5ad_path": os.environ.get("TCMIL_H5AD", _DEFAULT_H5AD),
        "metadata_csv_path": "/home/users/kcytsui/tcellMIL/data/Master_atlas_metadata.csv",
        "output_dir": os.environ.get("TCMIL_OUTDIR", _DEFAULT_OUT),

        # Autoencoder
        "latent_dim": 48,
        "ae_hidden_dim": 96,
        "ae_learning_rate": 1e-3,
        "ae_weight_decay": 1e-4,
        "ae_max_epochs": 200,
        "ae_patience": 15,
        "ae_batch_size": 256,
        "ae_dropout": 0.1,

        # MIL — 24c: anti-overfit knobs informed by diagnostic
        # (24a memorized to train_acc≥0.95 in 14% of folds; loss never plateaued
        #  through 100 epochs; LR schedule never fired). Three changes from 24a:
        # mil_dropout 0.2 → 0.4, mil_max_epochs 100 → 60, weight_decay 5e-2 → 1e-1.
        "hidden_dim": 64,
        "mil_learning_rate": 1e-3,
        "mil_weight_decay": 1e-1,
        "mil_max_epochs": 60,
        "mil_patience": 15,
        "mil_dropout": 0.4,
        "use_gated_attention": True,
        "num_classes": 2,

        # FiLM
        "film_hidden_dim": 32,

        # Class weight boost for minority class
        "minority_boost": 1.5,

        # No sample source encoding
        "use_sample_source": False,

        # WandB
        "wandb_project": "cohort_aware_antifit",
        "wandb_entity": "mackall_lab",
        "wandb_tags": ["response_prediction", "gated_attention", "no_sample_source",
                       "multiseed", "film", "age_sex"],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("\n" + "=" * 80)
    print("LOADING DATA (shared across all seeds)")
    print("=" * 80)
    adata = load_and_prepare_data(CONFIG)

    # Build demographic dict from CSV using the cohort patients
    cohort_patients = list(adata.obs["patient_id"].unique())
    if CONFIG.get("input_mode", "scenic") == "rna_pca":
        demo_dict, demo_dim, demo_stats = build_demo_dict_from_adata(
            adata, cohort_patients
        )
    else:
        demo_dict, demo_dim, demo_stats = build_demo_dict(
            CONFIG["metadata_csv_path"], cohort_patients
        )
    print(f"\nFiLM encoding (dim={demo_dim}): cohort one-hot {demo_stats['cohorts']}")
    print(f"  Patients: {demo_stats['n_patients']}")
    print(f"  Cohort counts: {demo_stats['cohort_counts']}")

    all_seed_results = {}
    seed_metrics = []

    for seed in SEEDS:
        print("\n" + "#" * 80)
        print(f"# SEED {seed}")
        print("#" * 80)

        results = run_loocv(CONFIG, adata, seed, demo_dict, demo_dim,
                            variant_name="cohort_aware_antifit")
        all_seed_results[seed] = results

        m = results["overall_metrics"]
        seed_metrics.append({
            "seed": seed,
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "auc": m["auc"],
        })

    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY")
    print("=" * 80)

    metrics_df = pd.DataFrame(seed_metrics)

    print(f"\n{'Seed':>6} {'Accuracy':>10} {'BalAcc':>10} {'AUC':>10} {'F1':>10}")
    print("-" * 50)
    for _, row in metrics_df.iterrows():
        print(f"{int(row['seed']):>6} {row['accuracy']:>10.4f} {row['balanced_accuracy']:>10.4f} "
              f"{row['auc']:>10.4f} {row['f1']:>10.4f}")
    print("-" * 50)

    for metric in ["accuracy", "balanced_accuracy", "auc", "f1"]:
        mean_val = metrics_df[metric].mean()
        std_val = metrics_df[metric].std()
        print(f"{metric:<20} {mean_val:.4f} +/- {std_val:.4f}")

    print(f"\nBaseline: Accuracy=0.714, AUC=0.725")
    mean_acc = metrics_df["accuracy"].mean()
    mean_auc = metrics_df["auc"].mean()
    print(f"Mean:     Accuracy={mean_acc:.4f}, AUC={mean_auc:.4f}")
    print(f"Delta:    Accuracy={mean_acc - 0.714:+.4f}, AUC={mean_auc - 0.725:+.4f}")

    combined_csv_path = os.path.join(CONFIG["output_dir"], "multiseed_summary.csv")
    metrics_df.to_csv(combined_csv_path, index=False)
    print(f"\nCombined metrics saved to: {combined_csv_path}")

    summary_row = {"seed": "mean_std"}
    for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"]:
        summary_row[metric] = f"{metrics_df[metric].mean():.4f}+/-{metrics_df[metric].std():.4f}"
    summary_df = pd.concat([metrics_df, pd.DataFrame([summary_row])], ignore_index=True)
    summary_csv_path = os.path.join(CONFIG["output_dir"], "multiseed_summary_with_stats.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary with stats saved to: {summary_csv_path}")

    all_pkl_path = os.path.join(CONFIG["output_dir"], "multiseed_all_results.pkl")
    with open(all_pkl_path, "wb") as f:
        pickle.dump(all_seed_results, f)
    print(f"All results pickle saved to: {all_pkl_path}")

    print("\n" + "=" * 80)
    print("MULTI-SEED PIPELINE COMPLETE")
    print("=" * 80)
