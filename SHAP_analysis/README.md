# Non-Directional SHAP Analysis for tcellMIL

## Overview

This repository contains the code for the non-directional (absolute magnitude) SHAP analysis linked to the submitted version of our AI4D3 workshop paper [TODO: insert link]. The analysis focuses on understanding transcription factor (TF) importance in patient-level predictions using Multiple Instance Learning (MIL) models trained on single-cell data.

## Methodology

### Non-Directional SHAP Approach

Instead of calculating directional SHAP values (positive or negative effects for patient classification), this analysis computes the **absolute magnitude** of SHAP contributions. This approach was chosen because:

- SHAP scores exhibit significant diversity across individual patients
- Using directional SHAP values would dilute the effect size when averaged across patients
- Absolute magnitude captures the overall importance of each transcription factor regardless of direction

### Analysis Pipeline

The analysis consists of two main components:

1. **Patient-Level SHAP Computation** (`Patient_level_SHAP.ipynb` and `patient_level_SHAP.py`):
   - Loads trained autoencoder and patient-specific MIL models
   - Creates patient bags from single-cell data
   - Applies SHAP gradient explainer to compute feature attributions
   - Generates both absolute and signed SHAP values for each patient
   - Focuses on positive class predictions (treatment response)

2. **Cross-Patient Analysis and Visualization** (`visualize_shap_results.py`):
   - Aggregates SHAP results across all patients
   - Computes consistency metrics and magnitude statistics
   - Creates comprehensive visualizations including:
     - Cross-patient importance heatmaps
     - Consistency vs magnitude scatter plots
     - Distribution plots for top transcription factors
     - Correlation analysis with attention weights


## Output

The analysis generates:
- Patient-specific SHAP importance rankings for 154 transcription factors
- Cross-patient summary statistics and visualizations
- Identification of most consistent and highest magnitude TFs
- Comparative analysis with model attention mechanisms


