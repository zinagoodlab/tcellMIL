# Cell Type Enrichment Analysis

This directory contains the analysis for investigating cell type enrichment patterns based on attention weights from trained tcellMIL model.

## Overview

The cell type enrichment analysis evaluates which cell types receive higher attention scores in the MIL model, potentially indicating their importance for treatment response prediction. The analysis uses permutation testing to determine statistical significance of attention score differences across cell types.

## Data Requirements

- **Single cell data**: `cell_atlas_axicel_IP_scenic_tf_matrix_added_v2.h5ad` - Annotated single cell expression data with patient metadata
- **MIL results**: `attention_weights/MIL_downstream_72_mil_results.pkl` - Pickled dictionary containing attention weights from trained MIL model
- **Required columns in single cell data**:
  - `patient_id`: Patient identifiers
  - `Response_3m`: Treatment response at 3 months
  - `cell_annotation`: Original cell type annotations
  - Attention scores are added during analysis

## Analysis Pipeline

### 1. Data Preprocessing
- Loads single cell data and removes patients with missing response data
- Loads MIL attention weights and adds them to single cell observations
- Groups cell types: combines "DC" and "Mono" into "Myeloid" category
- Filters out rare cell types (< 3 cells per patient)

### 2. Cell Type Enrichment Testing

#### Mean Attention Analysis
- **Method**: Permutation testing (1,000 permutations)
- **Metric**: Mean attention score per cell type per patient
- **Null hypothesis**: Cell type labels are randomly distributed
- **Test**: One-tailed empirical p-value (observed ≥ null)

#### Median Attention Analysis  
- **Method**: Permutation testing (10,000 permutations)
- **Metric**: Median attention score per cell type per patient
- **Purpose**: More robust to outliers than mean-based analysis

### 3. Statistical Analysis
- **Multiple testing correction**: Fisher's method for combining p-values across patients
- **FDR correction**: Applied to control false discovery rate
- **Patient-level analysis**: Separate enrichment scores calculated per patient
- **Response stratification**: Analysis split by treatment response status



