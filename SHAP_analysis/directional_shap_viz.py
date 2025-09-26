import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

def calculate_directional_metrics(signed_df):
    """Calculate various directional metrics that avoid dilution"""
    
    metrics = {}
    
    for tf in signed_df.index:
        values = signed_df.loc[tf]
        non_zero_mask = np.abs(values) > 0.001
        
        if non_zero_mask.sum() == 0:
            metrics[tf] = {
                'consensus_score': 0,
                'dominant_direction': 0,
                'magnitude_weighted_mean': 0,
                'directional_strength': 0,
                'consistency_ratio': 0
            }
            continue
            
        active_values = values[non_zero_mask]
        
        # 1. Consensus Score: Weighted mean by magnitude * directional consistency
        weights = np.abs(active_values)
        weighted_mean = np.average(active_values, weights=weights) if weights.sum() > 0 else 0
        
        positive_count = (active_values > 0).sum()
        negative_count = (active_values < 0).sum()
        total_count = len(active_values)
        consensus_factor = max(positive_count, negative_count) / total_count
        consensus_score = weighted_mean * consensus_factor
        
        # 2. Dominant Direction: Sign of majority * mean magnitude
        if positive_count > negative_count:
            dominant_direction = np.mean(active_values[active_values > 0])
        elif negative_count > positive_count:
            dominant_direction = np.mean(active_values[active_values < 0])
        else:
            dominant_direction = 0
            
        # 3. Magnitude-weighted mean (no consensus weighting)
        magnitude_weighted_mean = weighted_mean
        
        # 4. Directional Strength: How strong is the dominant direction
        if consensus_factor > 0.5:  # Clear majority
            majority_values = active_values[active_values > 0] if positive_count > negative_count else active_values[active_values < 0]
            directional_strength = np.mean(np.abs(majority_values)) if len(majority_values) > 0 else 0
            if negative_count > positive_count:
                directional_strength = -directional_strength
        else:
            directional_strength = 0
            
        # 5. Consistency Ratio: How consistent is the direction across patients
        consistency_ratio = consensus_factor
        
        metrics[tf] = {
            'consensus_score': consensus_score,
            'dominant_direction': dominant_direction,
            'magnitude_weighted_mean': magnitude_weighted_mean,
            'directional_strength': directional_strength,
            'consistency_ratio': consistency_ratio
        }
    
    return pd.DataFrame(metrics).T

def plot_directional_shap_attention_comparison(signed_df, attention_corr_df, method='consensus_score'):
    """
    Plot SHAP directionality vs attention correlation with various directional metrics
    
    Parameters:
    - method: 'consensus_score', 'dominant_direction', 'magnitude_weighted_mean', 'directional_strength'
    """
    
    if attention_corr_df is None:
        print("Cannot create plot - attention data not available")
        return
    
    # Calculate directional metrics
    directional_metrics = calculate_directional_metrics(signed_df)
    
    # Merge data
    common_tfs = set(directional_metrics.index) & set(attention_corr_df.index)
    plot_data = []
    
    for tf in common_tfs:
        plot_data.append({
            'TF': tf,
            'shap_directional': directional_metrics.loc[tf, method],
            'attention_correlation': attention_corr_df.loc[tf, 'correlation_kendall'],
            'attention_p_value': attention_corr_df.loc[tf, 'p_value_kendall'],
            'attention_neg_log_p': attention_corr_df.loc[tf, 'negative_log_p_value_kendall'],
            'consistency_ratio': directional_metrics.loc[tf, 'consistency_ratio']
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Color by consistency ratio (how reliable the direction is)
    scatter = plt.scatter(df_plot['attention_correlation'], df_plot['shap_directional'],
                         c=df_plot['consistency_ratio'], cmap='RdYlBu_r', 
                         s=80, edgecolors='white', linewidth=0.5, alpha=0.8,
                         vmin=0.5, vmax=1.0)
    
    plt.colorbar(scatter, label='Directional Consistency\n(Fraction agreeing on sign)')
    
    # Add quadrant lines
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add significance threshold lines
    plt.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=-0.1, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Correlation of Cell Attention and Regulon Activity\n(mean Kendall Tau)', fontsize=14)
    
    # Y-axis label depends on method
    method_labels = {
        'consensus_score': 'SHAP Consensus Score\n(Magnitude × Directional Agreement)',
        'dominant_direction': 'SHAP Dominant Direction\n(Majority Direction × Magnitude)',
        'magnitude_weighted_mean': 'SHAP Magnitude-Weighted Mean',
        'directional_strength': 'SHAP Directional Strength\n(Majority Effect Size)'
    }
    
    plt.ylabel(method_labels.get(method, f'SHAP {method}'), fontsize=14)
    plt.title(f'Directional SHAP vs Attention Correlation\n(Method: {method})', fontsize=16, pad=20)
    
    # Add text labels for significant/extreme TFs
    # Define thresholds based on the method
    if method in ['consensus_score', 'directional_strength']:
        y_threshold = np.abs(df_plot['shap_directional']).quantile(0.85)
    else:
        y_threshold = df_plot['shap_directional'].abs().quantile(0.85)
    
    x_threshold = 0.1
    consistency_threshold = 0.7
    
    texts = []
    for idx, row in df_plot.iterrows():
        if ((abs(row['shap_directional']) >= y_threshold or 
             abs(row['attention_correlation']) >= x_threshold) and
            row['consistency_ratio'] >= consistency_threshold):
            text = plt.text(row['attention_correlation'], row['shap_directional'], 
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

def plot_directional_comparison_grid(signed_df, attention_corr_df):
    """Plot a 2x2 grid comparing different directional methods"""
    
    if attention_corr_df is None:
        print("Cannot create plots - attention data not available")
        return
    
    methods = ['consensus_score', 'dominant_direction', 'magnitude_weighted_mean', 'directional_strength']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    directional_metrics = calculate_directional_metrics(signed_df)
    common_tfs = set(directional_metrics.index) & set(attention_corr_df.index)
    
    for i, method in enumerate(methods):
        ax = axes[i]
        
        # Prepare data
        plot_data = []
        for tf in common_tfs:
            plot_data.append({
                'TF': tf,
                'shap_directional': directional_metrics.loc[tf, method],
                'attention_correlation': attention_corr_df.loc[tf, 'correlation_kendall'],
                'consistency_ratio': directional_metrics.loc[tf, 'consistency_ratio']
            })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create scatter plot
        scatter = ax.scatter(df_plot['attention_correlation'], df_plot['shap_directional'],
                           c=df_plot['consistency_ratio'], cmap='RdYlBu_r', 
                           s=60, edgecolors='white', linewidth=0.5, alpha=0.8,
                           vmin=0.5, vmax=1.0)
        
        # Add reference lines
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-0.1, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Attention Correlation (Kendall Tau)', fontsize=12)
        ax.set_ylabel(method.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Method: {method.replace("_", " ").title()}', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label='Directional Consistency')
    
    plt.suptitle('Comparison of Directional SHAP Metrics vs Attention Correlation', 
                 fontsize=18, y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.show()

# Example usage functions
def main_directional_analysis(signed_df, attention_corr_df):
    """Run the directional analysis with all methods"""
    
    print("Creating directional SHAP vs attention plots...")
    
    # Method 1: Consensus Score (recommended)
    print("\n1. Consensus Score Method (magnitude × directional agreement)")
    df1 = plot_directional_shap_attention_comparison(signed_df, attention_corr_df, 'consensus_score')
    
    # Method 2: Dominant Direction
    print("\n2. Dominant Direction Method (majority direction × magnitude)")  
    df2 = plot_directional_shap_attention_comparison(signed_df, attention_corr_df, 'dominant_direction')
    
    # Method 3: Grid comparison
    print("\n3. Grid comparison of all methods")
    plot_directional_comparison_grid(signed_df, attention_corr_df)
    
    return df1, df2

