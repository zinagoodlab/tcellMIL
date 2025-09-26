import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc
import shap
import os
from torch.nn import functional as F


class PatientLevelModelWrapper(nn.Module):
    """Process patient-level bags for SHAP analysis"""
    
    def __init__(self, autoencoder, mil_model, sample_source_tensor):
        super().__init__()
        self.autoencoder = autoencoder
        self.mil_model = mil_model
        self.sample_source_tensor = sample_source_tensor
        
    def forward(self, patient_bag):
        """
        Forward pass for a single patient bag
        
        Args:
            patient_bag: [num_cells, num_features] - all cells for one patient
        Returns:
            logits: [num_classes] - prediction for this patient
        """
        # Encode all cells in the bag
        encoded_cells = self.autoencoder.encode(patient_bag)
        
        # MIL prediction (expects list of bags, so wrap in list)
        logits = self.mil_model([encoded_cells], self.sample_source_tensor)
        
        # Return single patient prediction
        return logits.squeeze(0)  # Remove batch dimension


def create_patient_bags(adata, patient_ids=None):
    """
    Create patient bags from AnnData object
    
    Args:
        adata: AnnData object with cell data
        patient_ids: List of patient IDs to process (if None, use all)
    
    Returns:
        dict: {patient_id: torch.Tensor of cells}
    """
    if patient_ids is None:
        patient_ids = adata.obs['patient_id'].unique()
    
    patient_bags = {}
    for patient_id in patient_ids:
        # Get cells for this patient
        patient_mask = adata.obs['patient_id'] == patient_id
        patient_cells = adata.X[patient_mask]
        
        # Convert to dense array if sparse
        if hasattr(patient_cells, 'toarray'):
            patient_cells = patient_cells.toarray()
            
        patient_bags[patient_id] = torch.FloatTensor(patient_cells)
    
    return patient_bags


def run_patient_level_shap(autoencoder_path, mil_models_dir, data_path, output_dir="shap_results"):
    """
    Run patient-level SHAP analysis
    
    Args:
        autoencoder_path: Path to trained autoencoder
        mil_models_dir: Directory containing MIL models for each patient
        data_path: Path to data file
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    adata = sc.read_h5ad(data_path)
    adata = adata[~adata.obs['Response_3m'].isna()].copy()
    adata.X = (adata.X - 0.5) * 2  # Match autoencoder preprocessing
    
    # Load shared autoencoder
    print("Loading autoencoder...")
    from src.Autoencoder import Autoencoder
    autoencoder = Autoencoder(input_dim=154, latent_dim=64)
    autoencoder.load_state_dict(torch.load(autoencoder_path, map_location="cpu"))
    autoencoder.eval()
    
    # Get sample source info
    tf_names = adata.var_names.tolist()
    source_list = list(adata.obs["Sample_source"].unique())
    
    # Get MIL model directories
    patient_dirs = [d for d in os.listdir(mil_models_dir) if os.path.isdir(os.path.join(mil_models_dir, d))]
    
    # Create patient bags for background (use subset of other patients)
    print("Creating background dataset...")
    all_patients = adata.obs['patient_id'].unique()
    
    for patient_folder in patient_dirs:
        patient_id = patient_folder.replace("patient_", "")
        model_path = os.path.join(mil_models_dir, patient_folder, "best_model.pth")
        
        if not os.path.exists(model_path):
            print(f"Skipping {patient_id} - model not found")
            continue
            
        print(f"Processing patient {patient_id}...")
        
        # Load MIL model
        from src.MIL import AttentionMIL
        mil_model = AttentionMIL(input_dim=64, num_classes=2, hidden_dim=128, sample_source_dim=4)
        mil_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        mil_model.eval()
        
        # Get test patient data
        test_patient_bag = create_patient_bags(adata, [patient_id])
        if patient_id not in test_patient_bag:
            print(f"No data found for patient {patient_id}")
            continue
            
        # Get sample source for this patient
        patient_sample_source = adata.obs[adata.obs['patient_id'] == patient_id]['Sample_source'].iloc[0]
        source_index = source_list.index(patient_sample_source)
        sample_onehot = torch.FloatTensor(np.eye(len(source_list))[source_index]).unsqueeze(0)
        
        # Create model wrapper
        wrapped_model = PatientLevelModelWrapper(autoencoder, mil_model, sample_onehot)
        
        # Create background dataset (other patients, max 10 for efficiency)
        background_patients = [p for p in all_patients if p != patient_id][:10]
        background_bags = create_patient_bags(adata, background_patients)
        
        # Stack background bags (use mean bag size to handle variable sizes)
        background_tensors = list(background_bags.values())
        min_cells = min(bag.shape[0] for bag in background_tensors)
        
        # Subsample each bag to common size
        background_data = []
        for bag in background_tensors:
            if bag.shape[0] > min_cells:
                indices = torch.randperm(bag.shape[0])[:min_cells]
                background_data.append(bag[indices])
            else:
                background_data.append(bag)
        
        background_tensor = torch.stack(background_data)  # [num_bg_patients, min_cells, features]
        
        # For SHAP, we need a representative background. Use mean across cells for each patient
        background_mean = background_tensor.mean(dim=1)  # [num_bg_patients, features]
        
        # Prepare test data (use full patient bag)
        test_bag = test_patient_bag[patient_id]
        
        # Run SHAP analysis
        print(f"Running SHAP for patient {patient_id}...")
        
        # Create a simplified wrapper that takes averaged patient representations
        class SimplifiedWrapper(nn.Module):
            def __init__(self, model_wrapper):
                super().__init__()
                self.model_wrapper = model_wrapper
                
            def forward(self, x):
                # x is [batch_size, features] - treat each row as a "patient"
                # For actual analysis, we need to expand back to bag format
                results = []
                for i in range(x.shape[0]):
                    # Create a single-cell "bag" for SHAP attribution
                    single_cell_bag = x[i].unsqueeze(0)  # [1, features]
                    result = self.model_wrapper(single_cell_bag)
                    results.append(result)
                return torch.stack(results)
        
        simple_wrapper = SimplifiedWrapper(wrapped_model)
        
        # Initialize SHAP explainer
        explainer = shap.GradientExplainer(simple_wrapper, background_mean)
        
        # Calculate SHAP values for representative cells from test patient
        test_sample_size = min(100, test_bag.shape[0])  # Sample up to 100 cells
        if test_bag.shape[0] > test_sample_size:
            indices = torch.randperm(test_bag.shape[0])[:test_sample_size]
            test_sample = test_bag[indices]
        else:
            test_sample = test_bag
            
        shap_values = explainer.shap_values(test_sample)
        
        # Debug: Print shapes for troubleshooting
        print(f"SHAP values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"SHAP values list length: {len(shap_values)}")
            for i, sv in enumerate(shap_values):
                print(f"SHAP values class {i} shape: {sv.shape}")
        else:
            print(f"SHAP values shape: {shap_values.shape}")
        
        # Process and save results - Focus on positive class only for binary classification
        if isinstance(shap_values, list):
            # Multi-class case - SHAP returns list of arrays
            shap_positive = shap_values[1]  # Positive class (OR)
        elif len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
            # Binary case with 3D output: (samples, features, classes)
            shap_positive = shap_values[:, :, 1]  # Positive class (OR)
        else:
            # Binary case with single output
            shap_positive = shap_values
        
        # Debug: Check shapes after processing
        print(f"Positive class SHAP shape: {shap_positive.shape}")
        
        # Aggregate SHAP values across cells
        mean_abs_shap = np.mean(np.abs(shap_positive), axis=0)
        mean_signed_shap = np.mean(shap_positive, axis=0)
        
        # Debug: Check shapes after aggregation
        print(f"mean_abs_shap shape: {mean_abs_shap.shape}")
        print(f"mean_signed_shap shape: {mean_signed_shap.shape}")
        print(f"tf_names length: {len(tf_names)}")
        
        # Ensure arrays are 1-dimensional for DataFrame creation
        mean_abs_shap = np.asarray(mean_abs_shap).flatten()
        mean_signed_shap = np.asarray(mean_signed_shap).flatten()
        
        # Create results DataFrames
        # Absolute importance (magnitude of contribution)
        df_abs = pd.DataFrame({
            'TF': tf_names,
            'mean_abs_SHAP': mean_abs_shap
        }).sort_values('mean_abs_SHAP', ascending=False)
        
        # Directional importance (signed contribution)
        df_signed = pd.DataFrame({
            'TF': tf_names,
            'mean_SHAP': mean_signed_shap
        }).sort_values('mean_SHAP', ascending=False)
        
        # Save results
        df_abs.to_csv(f"{output_dir}/{patient_folder}_patient_shap_positive_abs.csv", index=False)
        df_signed.to_csv(f"{output_dir}/{patient_folder}_patient_shap_positive_signed.csv", index=False)
        
        print(f"Completed SHAP analysis for patient {patient_id}")


if __name__ == "__main__":
    # Example usage
    autoencoder_path = "/Users/kristintsui/HA_MIL_model/tcellMIL/run_Sample_source_added_tcellMIL/autoencoder/best_autoencoder.pth"
    mil_models_dir = "/Users/kristintsui/HA_MIL_model/tcellMIL/run_Sample_source_added_tcellMIL/mil"
    data_path = "data/cell_atlas_axicel_IP_scenic_tf_matrix_added_v2.h5ad"
    
    run_patient_level_shap(autoencoder_path, mil_models_dir, data_path)
