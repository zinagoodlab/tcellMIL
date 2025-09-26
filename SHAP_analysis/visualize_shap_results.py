import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy import stats
from adjustText import adjust_text
import pickle
import scanpy as sc
from scipy.stats import kendalltau

def load_all_shap_data(shap_results_dir):
    """Load all SHAP results from the directory"""
    abs_files = glob.glob(os.path.join(shap_results_dir, "*_positive_abs.csv"))
    signed_files = glob.glob(os.path.join(shap_results_dir, "*_positive_signed.csv"))
    
    # Load absolute SHAP values
    abs_data = {}
    for file in abs_files:
        patient_id = os.path.basename(file).replace("_patient_shap_positive_abs.csv", "").replace("patient_", "")
        df = pd.read_csv(file)
        abs_data[patient_id] = df.set_index('TF')['mean_abs_SHAP']
    
    # Load signed SHAP values
    signed_data = {}
    for file in signed_files:
        patient_id = os.path.basename(file).replace("_patient_shap_positive_signed.csv", "").replace("patient_", "")
        df = pd.read_csv(file)
        signed_data[patient_id] = df.set_index('TF')['mean_SHAP']
    
    # Convert to DataFrames
    abs_df = pd.DataFrame(abs_data).fillna(0)
    signed_df = pd.DataFrame(signed_data).fillna(0)
    
    return abs_df, signed_df

def compute_attention_correlations(attention_file, data_file):
    """Compute correlations between attention weights and regulon activity"""
    try:
        # Load attention weights
        with open(attention_file, 'rb') as f:
            mil_results = pickle.load(f)

        
        
        # Load data
        adata = sc.read_h5ad(data_file)
        adata = adata[~adata.obs['Response_3m'].isna()].copy()
        adata.X = (adata.X - 0.5) * 2  # Match preprocessing
        
        # Extract attention weights and concatenate
        attn_weights = mil_results["attention_weights"]
        all_attns = np.concatenate([np.hstack(attn_weights[p]).flatten() for p in attn_weights.keys()])
        
        # Compute correlations for each TF
        correlations_kendall = []
        p_values_kendall = []
        tf_names = adata.var_names.tolist()
        
        for i in range(adata.X.shape[1]):
            gene_expr = np.array(adata.X[:, i]).flatten()
            corr_kendall, p_val_kendall = kendalltau(all_attns, gene_expr)
            correlations_kendall.append(corr_kendall)
            p_values_kendall.append(p_val_kendall)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'TF': tf_names,
            'correlation_kendall': correlations_kendall,
            'p_value_kendall': p_values_kendall,
            'negative_log_p_value_kendall': -np.log10(np.maximum(p_values_kendall, 1e-300))
        })
        
        return results.set_index('TF')
        
    except Exception as e:
        print(f"Could not load attention correlations: {e}")
        return None

def plot_shap_heatmap(abs_df, top_n=20):
    """Plot heatmap of top TFs across patients"""
    # Get top TFs by mean absolute SHAP across all patients
    mean_importance = abs_df.mean(axis=1).sort_values(ascending=False)
    top_tfs = mean_importance.head(top_n).index
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    heatmap_data = abs_df.loc[top_tfs].T
    
    sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Mean Absolute SHAP'})
    plt.title(f'Top {top_n} TF Importance Across Patients (SHAP)', fontsize=16, pad=20)
    plt.xlabel('Transcription Factors', fontsize=12)
    plt.ylabel('Patients', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return top_tfs

def plot_shap_consistency_vs_magnitude(abs_df, signed_df, min_patients=5):
    """Plot similar to correlation plot but for SHAP consistency vs magnitude"""
    
    # Calculate statistics for each TF
    tf_stats = []
    for tf in abs_df.index:
        abs_values = abs_df.loc[tf]
        signed_values = signed_df.loc[tf]
        
        # Only consider TFs that have non-zero SHAP in at least min_patients
        non_zero_mask = abs_values > 0.001  # Small threshold to handle numerical precision
        n_patients_with_signal = non_zero_mask.sum()
        
        if n_patients_with_signal >= min_patients:
            # Consistency: fraction of patients where this TF has importance
            consistency = n_patients_with_signal / len(abs_values)
            
            # Magnitude: mean absolute SHAP across all patients
            magnitude = abs_values.mean()
            
            # Direction consistency: how consistent is the direction?
            # Use coefficient of variation of signed values (only for non-zero)
            signed_nonzero = signed_values[non_zero_mask]
            if len(signed_nonzero) > 1:
                direction_consistency = 1 - (signed_nonzero.std() / abs(signed_nonzero.mean())) if signed_nonzero.mean() != 0 else 0
            else:
                direction_consistency = 1
            
            tf_stats.append({
                'TF': tf,
                'consistency': consistency,
                'magnitude': magnitude,
                'direction_consistency': direction_consistency,
                'n_patients': n_patients_with_signal
            })
    
    df_stats = pd.DataFrame(tf_stats)
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Color by direction consistency
    scatter = plt.scatter(df_stats['consistency'], df_stats['magnitude'], 
                         c=df_stats['direction_consistency'], 
                         cmap='viridis', s=80, 
                         edgecolors='white', linewidth=0.5, alpha=0.8)
    
    plt.colorbar(scatter, label='Direction Consistency')
    plt.xlabel('Consistency Across Patients (Fraction with Signal)', fontsize=14)
    plt.ylabel('Mean Absolute SHAP (Magnitude)', fontsize=14)
    plt.title('SHAP-based TF Importance: Consistency vs Magnitude', fontsize=16, pad=20)
    
    # Add text labels for top TFs
    # Select TFs that are high in both consistency and magnitude
    top_threshold_consistency = df_stats['consistency'].quantile(0.8)
    top_threshold_magnitude = df_stats['magnitude'].quantile(0.8)
    
    texts = []
    for idx, row in df_stats.iterrows():
        if (row['consistency'] >= top_threshold_consistency and 
            row['magnitude'] >= top_threshold_magnitude):
            text = plt.text(row['consistency'], row['magnitude'], 
                           row['TF'], fontsize=10, fontweight='bold')
            texts.append(text)
    
    # Adjust text positions to avoid overlaps
    if texts:
        adjust_text(texts, 
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7, linewidth=1),
                   expand_points=(1.2, 1.2))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return df_stats

def plot_top_tf_distributions(abs_df, signed_df, top_n=6):
    """Plot distribution of SHAP values for top TFs across patients"""
    
    # Get top TFs
    mean_importance = abs_df.mean(axis=1).sort_values(ascending=False)
    top_tfs = mean_importance.head(top_n).index
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, tf in enumerate(top_tfs):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get values for this TF
        abs_values = abs_df.loc[tf]
        signed_values = signed_df.loc[tf]
        
        # Create violin plot for absolute values
        parts = ax.violinplot([abs_values], positions=[1], widths=0.6, 
                             showmeans=True, showmedians=True)
        
        # Overlay scatter plot for individual patients
        y_jitter = np.random.normal(1, 0.05, len(abs_values))
        colors = ['red' if x < 0 else 'blue' for x in signed_values]
        ax.scatter(y_jitter, abs_values, c=colors, alpha=0.6, s=30)
        
        ax.set_title(f'{tf}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Absolute SHAP', fontsize=10)
        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([1])
        ax.set_xticklabels(['Patients'])
        ax.grid(True, alpha=0.3)
        
        # Add legend for the first subplot
        if i == 0:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                    markersize=8, label='Positive SHAP'),
                             Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                    markersize=8, label='Negative SHAP')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Remove empty subplots
    for i in range(len(top_tfs), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Distribution of SHAP Values for Top TFs Across Patients', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_shap_vs_attention_correlation(abs_df, attention_corr_df):
    """Plot SHAP importance vs attention correlation (similar to uploaded plot)"""
    
    if attention_corr_df is None:
        print("Cannot create SHAP vs attention plot - attention data not available")
        return
    
    # Calculate median SHAP importance across patients
    median_shap = abs_df.median(axis=1)
    
    # Merge data - only keep TFs present in both datasets
    common_tfs = set(median_shap.index) & set(attention_corr_df.index)
    plot_data = []
    
    for tf in common_tfs:
        plot_data.append({
            'TF': tf,
            'median_shap': median_shap[tf],
            'attention_correlation': attention_corr_df.loc[tf, 'correlation_kendall'],
            'attention_p_value': attention_corr_df.loc[tf, 'p_value_kendall'],
            'attention_neg_log_p': attention_corr_df.loc[tf, 'negative_log_p_value_kendall']
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Color by -log10(p-value) for attention correlation
    scatter = plt.scatter(df_plot['attention_correlation'], df_plot['median_shap'],
                         c=df_plot['attention_neg_log_p'], cmap='viridis', 
                         s=80, edgecolors='white', linewidth=0.5, alpha=0.8)
    
    plt.colorbar(scatter, label='-log₁₀(p-value)')
    plt.xlabel('Correlation of Cell Attention and Regulon Activity\n(mean Kendall Tau)', fontsize=14)
    plt.ylabel('Median SHAP Importance (Positive Class)', fontsize=14)
    plt.title('SHAP-based and Attention-based TF Importance', fontsize=16, pad=20)
    
    # Add significance lines 
    plt.axhline(y=df_plot['median_shap'].quantile(0.8), color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=-0.1, color='gray', linestyle='--', alpha=0.5)
    
    # Add text labels for significant TFs
    sig_threshold_p = 0.05
    shap_threshold = df_plot['median_shap'].quantile(0.8)
    corr_threshold = 0.1
    
    texts = []
    for idx, row in df_plot.iterrows():
        if (row['attention_p_value'] < sig_threshold_p and 
            (row['median_shap'] >= shap_threshold or abs(row['attention_correlation']) >= corr_threshold)):
            text = plt.text(row['attention_correlation'], row['median_shap'], 
                           row['TF'], fontsize=10, fontweight='bold')
            texts.append(text)
    
    # Adjust text positions 
    if texts:
        adjust_text(texts, 
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7, linewidth=1),
                   expand_points=(1.2, 1.2))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return df_plot

def main(attention_file=None, data_file=None):
    """Main function to generate all visualizations"""
    shap_results_dir = "shap_results"
    
    print("Loading SHAP data...")
    abs_df, signed_df = load_all_shap_data(shap_results_dir)
    print(f"Loaded data for {len(abs_df.columns)} patients and {len(abs_df.index)} TFs")
    
    # load attention correlations 
    attention_corr_df = None
    if attention_file and data_file:
        print("\nComputing attention correlations...")
        attention_corr_df = compute_attention_correlations(attention_file, data_file)
        if attention_corr_df is not None:
            print(f"Computed attention correlations for {len(attention_corr_df)} TFs")
    
    print("\n1. Creating cross-patient SHAP importance heatmap...")
    top_tfs = plot_shap_heatmap(abs_df, top_n=20)
    
    print("\n2. Creating SHAP consistency vs magnitude plot...")
    tf_stats = plot_shap_consistency_vs_magnitude(abs_df, signed_df, min_patients=5)
    
    print("\n3. Creating top TF distribution plots...")
    plot_top_tf_distributions(abs_df, signed_df, top_n=6)
    
    # Create SHAP vs attention comparison plot if data available
    if attention_corr_df is not None:
        print("\n4. Creating SHAP vs attention correlation plot...")
        comparison_df = plot_shap_vs_attention_correlation(abs_df, attention_corr_df)
        
        # Print correlation between SHAP and attention importance
        shap_attn_corr = stats.spearmanr(comparison_df['median_shap'], 
                                        comparison_df['attention_correlation'].abs())[0]
        print(f"Correlation between SHAP importance and attention correlation: {shap_attn_corr:.3f}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"- Most consistent TF: {tf_stats.loc[tf_stats['consistency'].idxmax(), 'TF']}")
    print(f"- Highest magnitude TF: {tf_stats.loc[tf_stats['magnitude'].idxmax(), 'TF']}")
    print(f"- Most directionally consistent TF: {tf_stats.loc[tf_stats['direction_consistency'].idxmax(), 'TF']}")

if __name__ == "__main__":
    main()
