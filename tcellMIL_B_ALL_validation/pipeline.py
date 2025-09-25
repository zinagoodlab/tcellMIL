import os
from datetime import datetime

from data_preprocessing import load_and_explore_data, preprocess_data
from autoencoder import train_autoencoder, evaluate_autoencoder


def create_latent(input_file, output_dir='results',
                       latent_dim=64, num_epochs_ae=200,
                       num_epochs=50, num_classes=2,
                       hidden_dim=128, 
                       project_name="car-t-response"):
    """run complete pipeline with leave one out cross validation
    
    Parameters:
    - input_file: path to input file
    - output_dir: directory to save results
    - latent_dim: dimension of latent space
    - num_epochs_ae: number of epochs for autoencoder
    - num_epoch_mil: number of epochs for MIL
    - num_classes: number of classes
    - hidden_dim: dimension of hidden layer

    Returns:
    - dict of results and models
    """

    # config = {
    #     "input_file": input_file,
    #     "output_dir": output_dir,
    #     "latent_dim": latent_dim,
    #     "num_epochs_ae": num_epochs_ae,
    #     "num_epochs_mil": num_epochs,
    #     "num_classes": num_classes,
    #     "hidden_dim": hidden_dim,
    #     "cv_method": "leave-one-out"
    # }

    # wandb.init(project=project_name, config=config)

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"run_{timestamp}")
    ae_dir = os.path.join(result_dir, "autoencoder")
    mil_dir = os.path.join(result_dir, "mil")
    
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ae_dir, exist_ok=True)
    os.makedirs(mil_dir, exist_ok=True)
    
    
    # Step 1: Load and explore data
    print("\n" + "="*80)
    print("STEP 1: LOADING AND EXPLORING DATA")
    print("="*80)
    adata = load_and_explore_data(input_file)



    # step 2: train autoencoder
    print("\n" + "="*80)
    print("STEP 2: TRAINING AUTOENCODER")
    print("="*80)
    train_loader, val_loader, test_loader, input_dim = preprocess_data(adata)

    # Step 3:train autoencoder
    print("\n" + "="*80)
    print("STEP 3: TRAINING AUTOENCODER")
    print("="*80)
    model, train_losses, val_losses = train_autoencoder(
            train_loader, val_loader, input_dim, latent_dim, num_epochs_ae, save_path=ae_dir
        )
    adata_latent, test_loss = evaluate_autoencoder(
        model, test_loader, adata, adata.var_names.tolist(), save_path=ae_dir
    )
    
    # Save latent representations
    latent_file = os.path.join(ae_dir, "latent_representation.h5ad")
    adata_latent.write(latent_file)

    return adata_latent

    # # Step 4: Run LOOCV
    # print("\n" + "="*80)
    # print("STEP 4: RUNNING LEAVE-ONE-OUT CROSS-VALIDATION")
    # print("="*80)
    
    # # Check if we have response information
    # if 'response' not in adata_latent.obs.columns:
    #     print("ERROR: 'response' column not found in the data. Cannot proceed with MIL.")
    #     # wandb.finish()
    #     return None
    
    # # Remove patients with NaN responses
    # patients_with_missing = adata_latent.obs[adata_latent.obs['response'].isna()]['patient_id'].unique()
    # if len(patients_with_missing) > 0:
    #     print(f"Removing {len(patients_with_missing)} patients with missing responses")
    #     adata_latent = adata_latent[~adata_latent.obs['patient_id'].isin(patients_with_missing)].copy()
        
    #     # update wandb config
    #     # wandb.config.update({
    #     #     "patients_after_filtering": adata_latent.obs['patient_id'].nunique(),
    #     #     "cells_after_filtering": adata_latent.n_obs,
    #     #     "patients_removed": len(patients_with_missing)
    #     #     })
        
    # cv_results = leave_one_out_cross_validation(
    #     adata_latent, 
    #     input_dim = latent_dim,
    #     num_classes = num_classes, 
    #     hidden_dim = hidden_dim,
    #     sample_source_dim = sample_source_dim,
    #     num_epochs = num_epochs,
    #     save_path = mil_dir
    # )
        


    # # wandb.finish()

    # print(f"Pipeline completed successfully! Results saved to {result_dir}")

    # return {
    #     'adata': adata,
    #     'autoencoder': model,
    #     'latent_data': adata_latent,
    #     'mil_results': cv_results,
    #     'results_dir': result_dir
    # }


def main(file_path, latent_dim=48, num_epochs=100):
    """
    Main function to run the entire pipeline
    
    Parameters:
    - file_path: Path to h5ad file with SCENIC results
    - latent_dim: Dimension of latent space
    - num_epochs: Number of training epochs
    """
    # Load and explore data
    adata = load_and_explore_data(file_path)
    
    # Extract TF names
    tf_names = adata.var_names.tolist()
    
    # Preprocess data
    train_loader, val_loader, test_loader, input_dim = preprocess_data(adata)
    
    # Train autoencoder
    model, train_losses, val_losses = train_autoencoder(
        train_loader, val_loader, input_dim, latent_dim, num_epochs
    )
    
    # Evaluate autoencoder
    adata_latent, test_loss = evaluate_autoencoder(model, test_loader, adata, tf_names)
    
    print("Autoencoder training and evaluation complete!")
    print(f"Final test loss: {test_loss:.6f}")
    print(f"Latent representation shape: {adata_latent.shape}")
    print("Latent representation saved for MIL implementation.")
    
    return model, adata_latent
