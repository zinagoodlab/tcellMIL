import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict


class PatientBagDataset(Dataset):
    """
    Dataset for Multiple Instance Learning where each bag contains cells from one patient
    """
    def __init__(self, adata, patient_col='patient_id', label_col='response_new', sample_source_vocab=None):
        """
        Initialize the MIL dataset
        
        Parameters:
        - adata: AnnData object with latent representations and patient information
        - patient_col: Column name for patient identifiers
        - label_col: Column name for patient response labels
        """
        self.adata = adata
        self.patient_col = patient_col
        self.label_col = label_col
        
        # Get unique patients
        self.patients = adata.obs[patient_col].unique()
        
        # Create a mapping of patient to label
        self.patient_to_label = dict(zip(
            adata.obs[patient_col], 
            adata.obs[label_col]
        ))

        #############################################
        if sample_source_vocab is None:
            sample_sources = list(adata.obs["Sample_source"].unique())
        else:
            sample_sources = list(sample_source_vocab)
        self.sample_sources = sample_sources
        self.sample_source_dim = len(self.sample_sources)


        # if "Sample_source" in adata.obs.columns:
            # Create one-hot encoding for Sample_source using unique values
            
        self.patient_metadata = {}
        
        # Get Sample_source for each patient
        patient_source_map = adata.obs.groupby(self.patient_col, observed=True)['Sample_source'].first().to_dict()
        
        # Create one-hot encoding for each patient
        for patient in self.patients:
            if patient in patient_source_map:
                source = patient_source_map[patient]
                one_hot = [1 if s == source else 0 for s in self.sample_sources]
                self.patient_metadata[patient] = one_hot
            


        ##############################################
        
        # Create patient bags
        self.patient_bags = {}
        self.patient_labels = {}
        
        for patient in self.patients:
            # Get indices for this patient
            indices = np.where(adata.obs[patient_col] == patient)[0]
            
            # Get features for this patient's cells
            patient_data = adata.X[indices]
            if isinstance(patient_data, np.ndarray) == False:
                patient_data = patient_data.toarray()
                
            self.patient_bags[patient] = patient_data
            
            # Get label for this patient
            # Assuming all cells from the same patient have the same label
            label = self.patient_to_label[patient]
            self.patient_labels[patient] = label
        
        # Convert patients to a list for indexing
        self.patient_list = list(self.patients)
    
    def __len__(self):
        """Return the number of bags (patients)"""
        return len(self.patient_list)
    
    def __getitem__(self, idx):
        """
        Get a patient bag
        
        Returns:
        - bag: Tensor of shape [num_instances, features] for the patient
        - label: Label for the patient
        - patient: Patient identifier
        """
        patient = self.patient_list[idx]
        bag = self.patient_bags[patient]
        label = self.patient_labels[patient]
        
        # Convert to proper format
        bag = torch.FloatTensor(bag)
        
        # Handle different label types
        if isinstance(label, str):
            # Map string labels to integers
            
            label_map = {"NR":0, "OR":1}
            label = label_map[label]
        
        label = torch.tensor(label, dtype=torch.long)

        if self.patient_metadata:
            one_hot_sample_source = torch.tensor(self.patient_metadata[patient], dtype=torch.float)
            return bag, label, patient, one_hot_sample_source
        # else:
        #     return bag, label, patient


class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning model as used in scMILD
    """
    def __init__(self, input_dim, num_classes=2, hidden_dim=128, dropout=0.25, sample_source_dim=None):
        """
        Initialize the MIL model
        
        Parameters:
        - input_dim: Dimension of the input features (output of autoencoder)
        - num_classes: Number of response classes
        - hidden_dim: Dimension of hidden layer
        - dropout: Dropout rate
        """
        super(AttentionMIL, self).__init__()

        self.use_sample_source = sample_source_dim is not None
        self.sample_source_dim = sample_source_dim
        
        # Feature extractor network
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        if self.use_sample_source:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim + sample_source_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim//2, num_classes)
            )
        else:
            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            ) #nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, sample_source=None, return_attention=False):
        """
        Forward pass
        
        Parameters:
        - x: Input bag of instances [batch_size, num_instances, features]
        - return_attention: Whether to return attention weights
        
        Returns:
        - logits: Class logits [batch_size, num_classes]
        - attention_weights: Attention weights if return_attention=True
        """
        device = next(self.parameters()).device
        # x shape: [batch_size, num_instances, features]
        batch_size = len(x)
        
        # Process each bag
        all_logits = []
        all_attention_weights = []
        
        for i in range(batch_size):
            instances = x[i]  # [num_instances, features]
            
            # Extract features from each instance
            instance_features = self.feature_extractor(instances)  # [num_instances, hidden_dim]
            
            # Calculate attention scores
            attention_scores = self.attention(instance_features)  # [num_instances, 1]
            attention_weights = F.softmax(attention_scores, dim=0)  # [num_instances, 1]
            
            # Calculate weighted average of instance features
            weighted_features = torch.sum(
                instance_features * attention_weights, dim=0
            )  # [hidden_dim]
            
            

            #########################################################
            ########## Add sample source if provided #################
            #########################################################
            if self.use_sample_source and sample_source is not None:
                
            # get the sample source for this patient
                sample_source_i = sample_source.squeeze(0)
                assert sample_source_i.shape[0] == self.sample_source_dim, \
                    f"Sample source dimension mismatch: {sample_source_i.shape[0]} != {self.sample_source_dim}"
                
                combined_features = torch.cat([weighted_features, sample_source_i], dim=0)
                # print(f"combined_features shape: {combined_features.shape}")
                logits = self.classifier(combined_features)
            
            else:
                logits = self.classifier(weighted_features)
                
            all_logits.append(logits)
            all_attention_weights.append(attention_weights)
        
        # Stack results
        logits = torch.stack(all_logits)  # [batch_size, num_classes]
        attention_weights = all_attention_weights  # List of [num_instances, 1]
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, num_classes=2):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        
        # Alpha for class balancing (higher weight for minority class)
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.tensor([1-alpha, alpha])  # [weight_for_NR, weight_for_OR]
        else:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
        # Get class probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class log probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the probability of the true class
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Calculate focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply alpha if specified
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        
        alpha_t = self.alpha.gather(0, targets)
        
        # Calculate focal loss
        focal_loss = alpha_t * focal_term * ce_loss
        
        return focal_loss.mean()
