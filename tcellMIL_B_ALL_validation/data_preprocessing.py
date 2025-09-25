import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def create_response_new_column(adata):
    """
    Create response_new column based on Response and BCA_group columns
    
    Logic:
    - CR -> OR
    - NR -> NR  
    - RL+ with BCA1/BCA2 -> NR
    - RL+ with BCA3 -> NR #OR
    - RL- with BCA1 -> NR
    - RL- with BCA2/BCA3 -> OR
    """
    
    # Define conditions and corresponding choices
    conditions = [
        # CR cases
        adata.obs['response'] == 'CR',
        
        # NR cases  
        adata.obs['response'] == 'NR',
        
        # RL+ cases
        (adata.obs['response'] == 'RL+') & (adata.obs['BCA group'].isin(['BCA1', 'BCA2'])),
        (adata.obs['response'] == 'RL+') & (adata.obs['BCA group'] == 'BCA3'),
        
        # RL- cases
        (adata.obs['response'] == 'RL-') & (adata.obs['BCA group'] == 'BCA1'),
        (adata.obs['response'] == 'RL-') & (adata.obs['BCA group'].isin(['BCA2', 'BCA3']))
    ]
    
    choices = ['OR', 'NR', 'NR', 'NR', 'NR', 'OR']
    
    adata.obs['response_new'] = np.select(conditions, choices, default='UNKNOWN')
    
    return adata


def load_and_explore_data(file_path):
    """
    Load the SCENIC AUC matrix from h5ad file and explore basic statistics as well as transform data range to [-1,1]
    """
    print("Loading SCENIC AUC matrix data...")
    adata = sc.read_h5ad(file_path)

    # subset to harad and deng datasets
    # adata = adata[adata.obs["Sample_source.y"] == "Deng"]
    adata = adata[~adata.obs['response_new'].isna()].copy()

    # shift adata.X to -0.5 to 0.5
    adata.X = (adata.X - 0.5) * 2

    # remove "Mono", "DC" and "other"
    
    # adata = adata[~adata.obs['cell_annotation'].isin(['Mono', 'DC', 'other'])].copy()
    # Print basic information
    print(f"Total cells: {adata.n_obs}")
    print(f"Number of TFs: {adata.n_vars}")
    
    # Check if we have patient information
    if 'patient_id' in adata.obs.columns:
        print(f"Number of patients: {adata.obs['patient_id'].nunique()}")
    
    # Check if we have response information
    if 'response_new' in adata.obs.columns:
        print("Response distribution:")
        print(adata.obs['response_new'].value_counts())
    
    # Basic stats on AUC values
    auc_matrix = adata.X
    if isinstance(auc_matrix, np.ndarray) == False:
        auc_matrix = auc_matrix.toarray()  # Convert sparse matrix if needed
    
    print(f"AUC matrix shape: {auc_matrix.shape}")
    print(f"AUC value range: [{np.min(auc_matrix)}, {np.max(auc_matrix)}]")
    print(f"AUC mean value: {np.mean(auc_matrix)}")
    
    return adata


def preprocess_data(adata, test_size=0.2, val_size=0.1, random_state=42):
    """Patient-level data split for autoencoder training
    
    Parameters:
    - adata: AnnData object with SCENIC results
    - test_size: Portion of data to use for testing
    - val_size: Portion of training data to use for validation
    - random_state: Random seed for reproducibility
    
    Returns:
    - train_loader: DataLoader for training
    - val_loader: DataLoader for validation
    - test_loader: DataLoader for testing
    - input_dim: Input dimension (number of TFs)
    """
    print("Preprocessing data for autoencoder training...")

    # get unique patients
    patients = adata.obs["patient_id"].unique()

    # split patients into train+val and test
    patients_train_val, patients_test = train_test_split(
        patients, test_size=test_size, random_state=random_state
    )

    # Further split train+val into train and val
    patients_train, patients_val = train_test_split(
        patients_train_val, test_size=val_size/(1-test_size), random_state=random_state
    )

    # select cells based on patient assignment
    train_mask = adata.obs["patient_id"].isin(patients_train)
    val_mask = adata.obs["patient_id"].isin(patients_val)
    test_mask = adata.obs["patient_id"].isin(patients_test)
    
    X_train = adata.X[train_mask]
    X_val = adata.X[val_mask]
    X_test = adata.X[test_mask]
    
    # Get AUC matrix and convert to dense if needed
    
    if isinstance(X_train, np.ndarray) == False:
        # X = X.toarray()
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()
    
    print(f"Training set: {X_train.shape[0]} cells")
    print(f"Validation set: {X_val.shape[0]} cells")
    print(f"Test set: {X_test.shape[0]} cells")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create DataLoaders (input and target are the same for autoencoders)
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, X_test_tensor)
    
    batch_size = 256  # Adjust based on your GPU memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    input_dim = X_train.shape[1]  # Number of TFs (should be 154)
    
    return train_loader, val_loader, test_loader, input_dim
