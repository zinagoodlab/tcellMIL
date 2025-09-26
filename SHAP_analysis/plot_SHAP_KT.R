

library(ggplot2)
library(dplyr)
library(ggrepel)

plot_shap_vs_attention_correlation <- function(shap_long_df, attention_corr_df) {
  # Check if attention data is available
  if (is.null(attention_corr_df)) {
    print("Cannot create SHAP vs attention plot - attention data not available")
    return(NULL)
  }
  
  # Calculate median SHAP importance across patients from long format
  library(dplyr)
  median_shap_df <- shap_long_df %>%
    group_by(TF) %>%
    summarise(median_shap = median(shap_value, na.rm = TRUE), .groups = 'drop')
  
  # Convert to named vector for easier indexing
  median_shap <- setNames(median_shap_df$median_shap, median_shap_df$TF)
  
  # Find common TFs present in both datasets
  # Use the TF names from the first column of attention_corr_df (which are row names)
  attention_tfs <- rownames(attention_corr_df)
  shap_tfs <- names(median_shap)
  common_tfs <- intersect(shap_tfs, attention_tfs)
  
  cat(sprintf("Found %d common TFs between SHAP and attention data\n", length(common_tfs)))
  
  # Create plot data
  plot_data <- data.frame(
    TF = common_tfs,
    median_shap = median_shap[common_tfs],
    attention_correlation = attention_corr_df[common_tfs, "correlation_kendall"],
    attention_p_value = attention_corr_df[common_tfs, "p_value_kendall"],
    attention_neg_log_p = -log10(attention_corr_df[common_tfs, "p_value_kendall_adjusted"]),
    stringsAsFactors = FALSE
  )
  
  # Define significance thresholds for labeling
  sig_threshold_p <- 0.05
  shap_threshold <- quantile(plot_data$median_shap, 0.8, na.rm = TRUE)
  corr_threshold <- 0.1
  
  # Create labels for significant TFs
  plot_data$label <- ifelse(
    plot_data$attention_p_value < sig_threshold_p & 
      (plot_data$median_shap >= shap_threshold | abs(plot_data$attention_correlation) >= corr_threshold),
    plot_data$TF, 
    ""
  )
  
  # Create the plot
  p <- ggplot(plot_data, aes(x = attention_correlation, y = median_shap)) +
    
    # Add reference lines
    geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
    geom_vline(xintercept = 0, color = "black", linewidth = 0.5) +
    
    # Add points
    geom_point(aes(fill = attention_neg_log_p), 
               shape = 21, color = "black", size = 2.5, stroke = 0.3, alpha = 0.8) +
    
    # Add text labels for all TFs (or subset based on your criteria)
    geom_text_repel(aes(label = TF), 
                    size = 3, 
                    max.overlaps = Inf,
                    box.padding = 0.3,
                    point.padding = 0.2,
                    segment.color = "transparent",
                    segment.alpha = 0,
                    # segment.linewidth = 0.3,
                    force = 2,
                    min.segment.length = 0) +
    
    # Color scale - red to gray gradient
    scale_fill_gradient(low = "gray60", high = "red", 
                        name = "-log₁₀(adj. p)",
                        na.value = "gray80") +
    
    # Clean theme
    theme_classic() +
    theme(
      plot.title = element_text(size = 14, hjust = 0.5, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9),
      panel.grid = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5),
      legend.key.size = unit(0.4, "cm")
    ) +
    
    # Labels
    labs(
      title = "Classification Importance vs. Cell Attention Importance",
      x = "Correlation of Cell Attention and Regulon Activity (mean Kendall Tau)",
      y = "Regulon SHAP Difference (OR vs. NR)"
    )
  
  # Print the plot
  print(p)
  
  # Return the plot data for further analysis
  return(plot_data)
}

plot_shap_vs_attention_correlation_selective <- function(shap_long_df, attention_corr_df, label_all = FALSE) {
  # Check if attention data is available
  if (is.null(attention_corr_df)) {
    print("Cannot create SHAP vs attention plot - attention data not available")
    return(NULL)
  }
  
  # Calculate median SHAP importance across patients from long format
  library(dplyr)
  median_shap_df <- shap_long_df %>%
    group_by(TF) %>%
    summarise(median_shap = median(shap_value, na.rm = TRUE), .groups = 'drop')
  
  # Convert to named vector for easier indexing
  median_shap <- setNames(median_shap_df$median_shap, median_shap_df$TF)
  
  # Find common TFs present in both datasets
  # Use the TF names from the first column of attention_corr_df (which are row names)
  attention_tfs <- rownames(attention_corr_df)
  shap_tfs <- names(median_shap)
  common_tfs <- intersect(shap_tfs, attention_tfs)
  
  cat(sprintf("Found %d common TFs between SHAP and attention data\n", length(common_tfs)))
  
  # Create plot data
  plot_data <- data.frame(
    TF = common_tfs,
    median_shap = median_shap[common_tfs],
    attention_correlation = attention_corr_df[common_tfs, "correlation_kendall"],
    attention_p_value = attention_corr_df[common_tfs, "p_value_kendall"],
    attention_neg_log_p = -log10(attention_corr_df[common_tfs, "p_value_kendall_adjusted"]),
    stringsAsFactors = FALSE
  )
  
  # Create labels - either all TFs or only significant ones
  if (label_all) {
    plot_data$label <- plot_data$TF
  } else {
    # Define significance thresholds for labeling
    sig_threshold_p <- 0.05
    shap_threshold <- quantile(plot_data$median_shap, 0.7, na.rm = TRUE)  # Adjust as needed
    corr_threshold <- 0.05  # Adjust as needed
    
    plot_data$label <- ifelse(
      plot_data$attention_p_value < sig_threshold_p | 
        abs(plot_data$median_shap) >= shap_threshold | 
        abs(plot_data$attention_correlation) >= corr_threshold,
      plot_data$TF, 
      ""
    )
  }
  
  # Create the plot
  p <- ggplot(plot_data, aes(x = attention_correlation, y = median_shap)) +
    
    # Add reference lines
    geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
    geom_vline(xintercept = 0, color = "black", linewidth = 0.5) +
    
    # Add points
    geom_point(aes(fill = attention_neg_log_p), 
               shape = 21, color = "black", size = 2.5, stroke = 0.3, alpha = 0.8) +
    
    # Add text labels
    geom_text_repel(aes(label = label), 
                    size = 3, 
                    max.overlaps = Inf,
                    box.padding = 0.2,
                    point.padding = 0.1,
                    segment.color = "transparent",
                    segment.alpha = 0,
                    # segment.linewidth = 0.3,
                    force = 1.5,
                    min.segment.length = 0.1) +
    
    # Color scale - red to gray gradient
    scale_fill_gradient(low = "gray60", high = "red", 
                        name = "-log₁₀(adj. p)",
                        na.value = "gray80") +
    
    # Clean theme
    theme_classic() +
    theme(
      plot.title = element_text(size = 14, hjust = 0.5, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9),
      panel.grid = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5),
      legend.key.size = unit(0.4, "cm")
    ) +
    
    # Labels to match the target plot
    labs(
      title = "Classification Importance vs. Cell Attention Importance",
      x = "Correlation of Cell Attention and Regulon Activity (mean Kendall Tau)",
      y = "Regulon SHAP Difference (OR vs. NR)"
    )
  
  # Print the plot
  print(p)
  
  # Return the plot data for further analysis
  return(plot_data)
}




# load datasets:
load_shap_from_csvs_long <- function(results_dir) {
  # Get only CSV files ending with _abs.csv
  csv_files <- list.files(results_dir, pattern = "*_signed\\.csv$", full.names = TRUE)
  
  if (length(csv_files) == 0) {
    stop("No CSV files ending with '_signed.csv' found in the specified directory")
  }
  
  cat(sprintf("Found %d files ending with '_signed.csv'\n", length(csv_files)))
  
  # Create long-form data list
  long_data_list <- list()
  
  for (i in seq_along(csv_files)) {
    file <- csv_files[i]
    
    # Extract patient ID from filename (remove _abs.csv)
    patient_id <- gsub("_signed\\.csv$", "", basename(file))
    
    cat(sprintf("Processing file %d/%d: %s (Patient: %s)\n", 
                i, length(csv_files), basename(file), patient_id))
    
    # Read the file
    tryCatch({
      shap_data <- read.csv(file, stringsAsFactors = FALSE)
      
      # Handle different possible file structures
      if (ncol(shap_data) >= 2) {
        # Assume first column is TF names, second is SHAP values
        # Adjust column names as needed
        if (all(c("TF", "shap_value") %in% colnames(shap_data))) {
          # Standard format
          tf_col <- "TF"
          shap_col <- "shap_value"
        } else if (all(c("gene", "importance") %in% colnames(shap_data))) {
          # Alternative format
          tf_col <- "gene"
          shap_col <- "importance"
        } else {
          # Use first two columns
          tf_col <- colnames(shap_data)[1]
          shap_col <- colnames(shap_data)[2]
          cat(sprintf("Using columns: %s (TF) and %s (SHAP)\n", tf_col, shap_col))
        }
        
        # Create long format data for this patient
        patient_data <- data.frame(
          patient_id = patient_id,
          TF = shap_data[[tf_col]],
          shap_value = as.numeric(shap_data[[shap_col]]),
          stringsAsFactors = FALSE
        )
        
      } else if (ncol(shap_data) == 1 && !is.null(rownames(shap_data))) {
        # Single column with row names as TFs
        patient_data <- data.frame(
          patient_id = patient_id,
          TF = rownames(shap_data),
          shap_value = as.numeric(shap_data[, 1]),
          stringsAsFactors = FALSE
        )
      } else {
        stop(sprintf("Cannot parse file structure for %s", basename(file)))
      }
      
      # Remove rows with missing TF names or SHAP values
      patient_data <- patient_data[!is.na(patient_data$TF) & 
                                     !is.na(patient_data$shap_value) &
                                     patient_data$TF != "", ]
      
      cat(sprintf("  Loaded %d TFs for patient %s\n", nrow(patient_data), patient_id))
      
      # Add to list
      long_data_list[[patient_id]] <- patient_data
      
    }, error = function(e) {
      cat(sprintf("Error processing file %s: %s\n", basename(file), e$message))
    })
  }
  
  # Combine all patient data into long format
  long_data <- do.call(rbind, long_data_list)
  
  if (nrow(long_data) == 0) {
    stop("No data was successfully loaded from any files")
  }
  
  cat(sprintf("Combined long data: %d rows total\n", nrow(long_data)))
  cat(sprintf("Unique patients: %d\n", length(unique(long_data$patient_id))))
  cat(sprintf("Unique TFs: %d\n", length(unique(long_data$TF))))
  
  return(long_data)
}

signed_df <- load_shap_from_csvs_long("/Users/kristintsui/HA_MIL_model/NIPS2025_Supplementary/code/shap_results")

abs_df <- load_shap_from_csvs_long("/Users/kristintsui/HA_MIL_model/NIPS2025_Supplementary/code/shap_results")

attention_corr_df <- read.csv("/Users/kristintsui/HA_MIL_model/NIPS2025_Supplementary/code/kendall_tau_attention_correlation_df.csv", row.names = "TF")

# 



library(ggplot2)
library(dplyr)
library(ggrepel)

plot_shap_vs_attention_correlation <- function(abs_df, attention_corr_df) {
  
  # Calculate median SHAP importance across patients 
  median_shap_df <- abs_df %>%
    group_by(TF) %>%
    summarise(median_shap = median(shap_value, na.rm = TRUE), .groups = 'drop')
  
  # Convert to named vector for easier indexing
  median_shap <- setNames(median_shap_df$median_shap, median_shap_df$TF)
  
  # Find common TFs 
  common_tfs <- intersect(names(median_shap), rownames(attention_corr_df))
  
  # Handle extreme p-values
  p_values <- attention_corr_df[common_tfs, "p_value_kendall"]
  p_values_adjusted <- p.adjust(p_values, method = "BH")
  
  # Replace 0 or very small p-values 
  min_pvalue <- 1e-300  # Or use the smallest non-zero p-value
  p_values_adjusted <- pmax(p_values_adjusted, min_pvalue)
  
  neg_log10_p <- -log10(p_values_adjusted)
  max_neg_log10 <- 300  # Cap at 300 (equivalent to p-value of 1e-300)
  neg_log10_p <- pmin(neg_log10_p, max_neg_log10)
  
  # Create the plotting dataframe
  plot_df <- data.frame(
    TF = common_tfs,
    mean_tau = attention_corr_df[common_tfs, "correlation_kendall"],
    estimate = median_shap[common_tfs],
    neg_log10_padj = neg_log10_p,
    stringsAsFactors = FALSE
  )
  

  fig4b_plot <- ggplot(plot_df, aes(x = mean_tau, y = estimate, label = TF)) +
    geom_point(aes(fill = neg_log10_padj), shape = 21, color = "black", size = 2, stroke = 0.5, alpha = 0.8) +
    geom_text_repel(max.overlaps = 35, size = 5) +
    geom_vline(xintercept = 0, color = "black") +
    geom_hline(yintercept = 0, color = "black") +
    scale_fill_gradient2(low = "blue", mid = "gray40", high = "red",
                         midpoint = median(plot_df$neg_log10_padj, na.rm = TRUE),
                         name = "-log10(p_adj)") +
    theme_classic() +
    labs(
      title = "Classification Importance vs. Cell Attention Importance",
      x = "Correlation of Cell Attention and Regulon Activity (mean Kendall Tau)",
      y = "Median SHAP Importance (Positive Class)"
    )
  
  print(fig4b_plot)
  return(plot_df)
}

# Example usage:
# abs data
plot_data <- plot_shap_vs_attention_correlation(abs_df, attention_corr_df)

# signed data
plot_signed_data <- plot_shap_vs_attention_correlation(signed_df, attention_corr_df)
ggsave("fig4b_plot_mean_kendall_and_median_shap_importance.pdf", plot_data, path = "/Users/kristintsui/HA_MIL_model/neurips_ai4d3_2025/Figures", width = 8, height = 8)
