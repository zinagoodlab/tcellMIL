import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
import anndata
import pickle

from MIL import PatientBagDataset, AttentionMIL, FocalLoss




def k_fold_cross_validation(adata, input_dim, num_classes=2, hidden_dim=128, 
                                       num_epochs=50, learning_rate=5e-4, response_col='response_og',
                                       weight_decay=1e-2, n_folds=3,  
                                       use_focal_loss=True, alpha=0.75, gamma=2.0,
                                       save_path='results'):
    """
    K-fold CV with SMOTE and Focal Loss for extreme imbalance.
    
    Parameters:
    - n_folds: Number of folds (use 3 for extreme imbalance)
    - use_focal_loss: Whether to use Focal Loss instead of CE
    - alpha: Weight for minority class in Focal Loss (0.75 = 75% weight on NR class)
    - gamma: Focusing parameter for Focal Loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(save_path, exist_ok=True)
    
    # set sample source dim
    sample_source_vocab = list(adata.obs['Sample_source'].unique())
    source_col = 'Sample_source'

    # Get original patients and labels

    full_dataset = PatientBagDataset(adata, label_col=response_col)
    print('using label col', full_dataset.label_col)

    original_patients = np.array(full_dataset.patient_list)
    original_labels = np.array([full_dataset.patient_labels[p] for p in original_patients])
    
    if isinstance(original_labels[0], str):
        original_labels = np.array([1 if l == "OR" else 0 for l in original_labels])
    
    print(f"Original distribution: OR={sum(original_labels==1)}, NR={sum(original_labels==0)}")
    
    
    # Use StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Storage for results
    all_true_labels = []
    all_predictions = []
    all_probabilities = []
    all_patient_ids = []
    fold_metrics = []

    patients = original_patients
    labels = original_labels
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(patients, labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*50}")
        
        train_patients = patients[train_idx]
        test_patients = patients[test_idx]
        
 
        
        # Print fold distribution
        train_labels = labels[train_idx]
        test_labels = labels[[i for i, p in enumerate(patients) if p in test_patients]]
        print(f"Train: OR={sum(train_labels==1)}, NR={sum(train_labels==0)}")
        print(f"Test: OR={sum(test_labels==1)}, NR={sum(test_labels==0)}")
        
        # Create datasets
        train_mask = adata.obs['patient_id'].isin(train_patients)
        test_mask = adata.obs['patient_id'].isin(test_patients)
        
        train_dataset = PatientBagDataset(adata[train_mask].copy(), 
                                          label_col=response_col,
                                          sample_source_vocab=sample_source_vocab)
        test_dataset = PatientBagDataset(adata[test_mask].copy(), 
                                         label_col=response_col,
                                         sample_source_vocab=sample_source_vocab)
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # Initialize model
        model = AttentionMIL(input_dim, num_classes, hidden_dim, sample_source_dim=len(sample_source_vocab)).to(device)
        
        # Choose loss function
        if use_focal_loss:
            # Calculate alpha based on class distribution in training
            n_nr = sum(train_labels == 0)
            n_or = sum(train_labels == 1)
            # Higher alpha for minority class
            alpha_value = [n_or/(n_nr+n_or), n_nr/(n_nr+n_or)]  # [weight_for_0, weight_for_1]
            criterion = FocalLoss(alpha=alpha_value, gamma=gamma, num_classes=num_classes).to(device)
            print(f"Using Focal Loss with alpha={alpha_value}, gamma={gamma}")
        else:
            # Standard weighted CE
            class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            class_weights = torch.FloatTensor(class_weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using Weighted CE with weights={class_weights.cpu().numpy()}")
        
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Training
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for bags, batch_labels, _, one_hot_sample_source in train_loader:
                bags = [bag.to(device) for bag in bags]
                batch_labels = batch_labels.to(device)
                one_hot_sample_source = one_hot_sample_source.to(device)
                
                optimizer.zero_grad()
                logits = model(bags, one_hot_sample_source)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, preds = torch.max(logits, 1)
                train_correct += (preds == batch_labels).sum().item()
                train_total += batch_labels.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}")
            
            scheduler.step(avg_train_loss)
            
            # Early stopping
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(save_path, f'best_model_fold_{fold_idx}.pth'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Evaluation
        # model.load_state_dict(torch.load(os.path.join(save_path, f'best_model_fold_{fold_idx}.pth')))
        model.eval()
        
        fold_predictions = []
        fold_labels = []
        fold_probs = []
        fold_patients = []
        
        with torch.no_grad():
            for bags, batch_labels, patient_ids, one_hot_sample_source in test_loader:
                bags = [bag.to(device) for bag in bags]
                batch_labels = batch_labels.to(device)
                one_hot_sample_source = one_hot_sample_source.to(device)
                
                logits, _ = model(bags, one_hot_sample_source, return_attention=True)
                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)
                
                fold_predictions.extend(preds.cpu().numpy())
                fold_labels.extend(batch_labels.cpu().numpy())
                fold_probs.extend(probs.cpu().numpy()[:, 1] if num_classes == 2 else probs.cpu().numpy())
                fold_patients.extend(patient_ids)
        
        # Store fold results
        all_true_labels.extend(fold_labels)
        all_predictions.extend(fold_predictions)
        all_probabilities.extend(fold_probs)
        all_patient_ids.extend(fold_patients)
        
        # Calculate fold metrics
        fold_acc = accuracy_score(fold_labels, fold_predictions)
        fold_prec = precision_score(fold_labels, fold_predictions, zero_division=0)
        fold_rec = recall_score(fold_labels, fold_predictions, zero_division=0)
        fold_f1 = f1_score(fold_labels, fold_predictions, zero_division=0)
        
        print(f"\nFold {fold_idx+1} Results:")
        print(f"Accuracy: {fold_acc:.4f}, Precision: {fold_prec:.4f}, Recall: {fold_rec:.4f}, F1: {fold_f1:.4f}")
        
        fold_metrics.append({
            'fold': fold_idx,
            'accuracy': fold_acc,
            'precision': fold_prec,
            'recall': fold_rec,
            'f1': fold_f1
        })
    
    # Calculate overall metrics
    overall_acc = accuracy_score(all_true_labels, all_predictions)
    overall_prec = precision_score(all_true_labels, all_predictions, zero_division=0)
    overall_rec = recall_score(all_true_labels, all_predictions, zero_division=0)
    overall_f1 = f1_score(all_true_labels, all_predictions, zero_division=0)
    overall_auc = roc_auc_score(all_true_labels, all_probabilities) if len(np.unique(all_true_labels)) > 1 else 0
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)
    
    print(f"\n{'='*50}")
    print("OVERALL RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {overall_acc:.4f}")
    print(f"Precision: {overall_prec:.4f}")
    print(f"Recall: {overall_rec:.4f}")
    print(f"F1 Score: {overall_f1:.4f}")
    print(f"AUC: {overall_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(conf_matrix)
    
    return {
        'overall_metrics': {
            'accuracy': overall_acc,
            'precision': overall_prec,
            'recall': overall_rec,
            'f1': overall_f1,
            'auc': overall_auc,
            'confusion_matrix': conf_matrix
        },
        'fold_metrics': fold_metrics,
        'patient_predictions': {
            pid: {'true': true, 'pred': pred, 'prob': prob}
            for pid, true, pred, prob in zip(all_patient_ids, all_true_labels, all_predictions, all_probabilities)
        }
    }


def leave_one_out_cross_validation(adata, input_dim, num_classes=2, hidden_dim=128, sample_source_dim=4, response_col='response_og',
                                  num_epochs=50, learning_rate=5e-4, weight_decay = 1e-2, alpha=0.75, gamma=2.0, use_focal_loss=True,
                                  save_path='results'):
    """
    Perform leave-one-out cross-validation for the MIL model
    # Parameters:
    - adata: AnnData object with latent representations
    - input_dim: Input dimension (latent space dimension)
    - num_classes: Number of response classes
    - hidden_dim: Dimension of hidden layer
    - num_epochs: Maximum number of training epochs
    - learning_rate: Learning rate for optimizer
    - weight_decay: L2 regularization strength

    Returns:
    - cv_results: Dictionary of cross-validation results
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(save_path, exist_ok=True)

    # initiate wandb
    # wandb.init(project="car-t-IP-MIL", name="loocv-mil",
    #            config={
    #                "input_dim": input_dim,
    #                "num_classes": num_classes,
    #                "hidden_dim": hidden_dim,
    #                "num_epochs": num_epochs,
    #                "learning_rate": learning_rate,
    #                "weight_decay": weight_decay,
    #                "save_path": save_path
    #            })
    
    # create dataset
    full_dataset = PatientBagDataset(adata)

    # get all patients and their labels
    patients = np.array(full_dataset.patient_list)
    labels = np.array([full_dataset.patient_labels[p] for p in patients])

    print(f"Performing leave-one-out cross-validation for {len(patients)} patients...")

    cv_results = {
        'fold_metrics': [], # store per-fold metrics for reference
        'patient_predictions': {},
        'attention_weights': {}
    }

    # Initialize accumulators for all predictions across folds
    all_true_labels = []
    all_predicted_labels = []
    all_prediction_probs = []
    all_patient_ids = []

    # wandb_patient_table = wandb.Table(columns=["patient_id", "true_label", "predicted_label", "predicted_label", "correct"])

    # all_metrics = []
    # all_confusion_matrices = np.zeros((num_classes, num_classes))


    # LOOCV loop
    for i, test_patient in enumerate(patients):

        print(f"Fold {i+1}/{len(patients)} patients, testing on {test_patient}...")
        train_patients = np.array([p for p in patients if p != test_patient])

        sample_source_vocab = list(adata.obs['Sample_source'].unique())
        
        # Create datasets
        train_mask = adata.obs['patient_id'].isin(train_patients)
        test_mask = adata.obs['patient_id'] == test_patient
        
        train_dataset = PatientBagDataset(adata[train_mask].copy(), 
                                          label_col=response_col,
                                          sample_source_vocab=sample_source_vocab)
        test_dataset = PatientBagDataset(adata[test_mask].copy(), 
                                         label_col=response_col,
                                         sample_source_vocab=sample_source_vocab)
        
        

        # create data loaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # create save path for this fold
        fold_save_path = os.path.join(save_path, f'patient_{test_patient}')
        os.makedirs(fold_save_path, exist_ok=True)

        # train model
        model = AttentionMIL(input_dim, num_classes, hidden_dim, sample_source_dim=len(sample_source_vocab)).to(device)

        
        # Get training labels only (not all labels)
        train_patients_set = set(train_patients)
        train_mask = adata.obs['patient_id'].isin(train_patients_set)
        y_train = adata.obs.loc[train_mask, 'response_col'].to_numpy()
        y_train = np.where(y_train == "NR", 0, 1)
        
        # Check if both classes are present in training data
        # unique_classes = np.unique(y_train)
        # if len(unique_classes) == 2:
        #     # Both classes present, use balanced weights
        #     class_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
        #     class_weights = torch.FloatTensor(class_weights).to(device)
        # else:
        #     # Only one class present, use standard CrossEntropyLoss
        #     print(f"Warning: Only one class present in training data for patient {test_patient}. Using standard loss.")
        #     class_weights = None
        
        # # Loss function and optimizer
        # if class_weights is not None:
        #     criterion = nn.CrossEntropyLoss(weight=class_weights)
        # else:
        #     criterion = nn.CrossEntropyLoss()


        # Choose loss function
        if use_focal_loss:
            # Calculate alpha based on class distribution in training
            n_nr = sum(y_train == 0)
            n_or = sum(y_train == 1)
            # Higher alpha for minority class
            alpha_value = [n_or/(n_nr+n_or), n_nr/(n_nr+n_or)]  # [weight_for_0, weight_for_1]
            criterion = FocalLoss(alpha=alpha_value, gamma=gamma, num_classes=num_classes).to(device)
            print(f"Using Focal Loss with alpha={alpha_value}, gamma={gamma}")
        else:
            # Standard weighted CE
            class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            class_weights = torch.FloatTensor(class_weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using Weighted CE with weights={class_weights.cpu().numpy()}")

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    

        # criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        
        # Training setup
        best_train_loss = float("inf")
        epochs_without_improvement = 0
        patience = 8

        history = {
            'train_loss': [],
            'train_acc': []
            
        }

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for bags, batch_labels, _, one_hot_sample_source in train_loader:
                bags = [bag.to(device) for bag in bags]
                batch_labels = batch_labels.to(device)
                one_hot_sample_source = one_hot_sample_source.to(device)

                # Forward pass
                logits = model(bags, one_hot_sample_source)
                loss = criterion(logits, batch_labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update metrics
                train_loss += loss.item()
                _, preds = torch.max(logits.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (preds == batch_labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0

            # record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)


            # use training loss for scheduler and early stopping since we have no validation set
            scheduler.step(train_loss)

            # print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f'Patient {test_patient} - Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

            # log metrics
            # wandb.log({
            #     f"patient_{test_patient}/epoch": epoch + 1,
            #     f"patient_{test_patient}/train_loss": train_loss,
            #     f"patient_{test_patient}/train_acc": train_acc,
            #     f"patient_{test_patient}/learning_rate": optimizer.param_groups[0]['lr'],
            # })

            # early stopping
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                epochs_without_improvement = 0

                #save best model
                torch.save(model.state_dict(), os.path.join(fold_save_path, 'best_model.pth'))

            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f'Early stopping after {epoch+1} epochs')
                    break
        
        # load best model
        model.load_state_dict(torch.load(os.path.join(fold_save_path, 'best_model.pth')))

        # evaluate on test patient
        model.eval()
        device = next(model.parameters()).device

        # test_preds = []
        # test_labels = []
        # test_probs = []
        # patient_attentions = {}

        with torch.no_grad():
            for bags, batch_labels, patient_ids, one_hot_sample_source in test_loader:
                bags = [bag.to(device) for bag in bags]
                batch_labels = batch_labels.to(device)
                one_hot_sample_source = one_hot_sample_source.to(device)

                # Forward pass
                logits, attn_weights = model(bags, one_hot_sample_source, return_attention=True)

                # get predictions
                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)

                # convert to numpy
                preds_np = preds.cpu().numpy()
                labels_np = batch_labels.cpu().numpy()
                probs_np = probs.cpu().numpy()

                # Get the patient_id and store results
                patient_id = patient_ids[0]
                true_label = labels_np[0]
                pred_label = preds_np[0]
                pos_prob = probs_np[0, 1] if num_classes == 2 else None

                # Add to accumulators for global metrics
                all_true_labels.append(true_label)
                all_predicted_labels.append(pred_label)
                all_prediction_probs.append(pos_prob if num_classes ==2 else probs_np[0])
                all_patient_ids.append(patient_id)

                # Store patient-specific results
                cv_results['patient_predictions'][patient_id] = {
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'probabilities': probs_np[0].tolist(),
                    'correct': (pred_label == true_label)
                }
                
                    
                

                # Store attention weights
                cv_results['attention_weights'][patient_id] = [w.cpu().numpy() for w in attn_weights]

                # Add to wandb table
                # wandb_patient_table.add_data(
                #     patient_id, 
                #     int(true_label), 
                #     int(pred_label), 
                #     float(pos_prob) if num_classes == 2 else 'N/A',
                #     bool(pred_label == true_label)
                # )

                
        # Calculate per-fold metrics for individual patient (for monitoring only)
        fold_correct = (preds_np[0] == labels_np[0])
        fold_metrics = {
            'patient_id': patient_id,
            'fold': i,
            'accuracy': 1.0 if fold_correct else 0.0,
            'true_label': int(labels_np[0]),
            'predicted_label': int(preds_np[0]),
            'prob_positive': float(probs_np[0, 1]) if num_classes == 2 else None
        }
        cv_results['fold_metrics'].append(fold_metrics)
        
        # Log patient result to wandb
        # wandb.log({
        #     f"patient_{patient_id}/true_label": labels_np[0],
        #     f"patient_{patient_id}/predicted_label": preds_np[0],
        #     f"patient_{patient_id}/prob_positive": probs_np[0, 1] if num_classes == 2 else None,
        #     f"patient_{patient_id}/correct": fold_correct
        # })

    # Convert accumulators to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_predicted_labels = np.array(all_predicted_labels)
    all_prediction_probs = np.array(all_prediction_probs)
    all_patient_ids = np.array(all_patient_ids)
    
    # Calculate overall metrics only once using ALL predictions
    overall_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    
    # For binary classification
    if num_classes == 2:
        overall_precision = precision_score(all_true_labels, all_predicted_labels)
        overall_recall = recall_score(all_true_labels, all_predicted_labels)
        overall_f1 = f1_score(all_true_labels, all_predicted_labels)
        overall_auc = roc_auc_score(all_true_labels, all_prediction_probs)
    else:
        # For multiclass classification
        overall_precision = precision_score(all_true_labels, all_predicted_labels, average='weighted')
        overall_recall = recall_score(all_true_labels, all_predicted_labels, average='weighted')
        overall_f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
        # For multiclass AUC, we'd need to calculate it differently (beyond scope here)
        overall_auc = 0.5  
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
    
    # Calculate class-specific metrics
    class_metrics = {}
    for c in range(num_classes):
        true_positives = np.sum((all_true_labels == c) & (all_predicted_labels == c))
        actual_positives = np.sum(all_true_labels == c)
        predicted_positives = np.sum(all_predicted_labels == c)
        
        class_metrics[f'class_{c}'] = {
            'precision': true_positives / predicted_positives if predicted_positives > 0 else 0,
            'recall': true_positives / actual_positives if actual_positives > 0 else 0,
            'count': int(actual_positives)
        }
    
    # Store final metrics in results
    cv_results['overall_metrics'] = {
        'accuracy': float(overall_accuracy),
        'precision': float(overall_precision),
        'recall': float(overall_recall),
        'f1': float(overall_f1),
        'auc': float(overall_auc),
        'confusion_matrix': conf_matrix.tolist(),
        'class_metrics': class_metrics
    }
    
    # Log final results to wandb
    # wandb.log({
    #     "overall_accuracy": overall_accuracy,
    #     "overall_precision": overall_precision,
    #     "overall_recall": overall_recall,
    #     "overall_f1": overall_f1,
    #     "overall_auc": overall_auc,
    #     "patient_results": wandb_patient_table
    # })
    
    # Print final results
    print("\n===== LOOCV Final Results =====")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({np.sum(all_predicted_labels == all_true_labels)}/{len(all_true_labels)} patients correct)")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"Overall AUC: {overall_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClass-specific Metrics:")
    for class_name, metrics in class_metrics.items():
        print(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, Count={metrics['count']}")
    
    # Save results to file
    results_path = os.path.join(save_path, 'loocv_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(cv_results, f)
    
    return cv_results
