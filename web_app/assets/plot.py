import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the detector stage CSV
CSV = "../../eval/detector_stage/detailed_analysis.csv"  # Adjust path as needed
df = pd.read_csv(CSV)

# Filter for test sets only
test_df = df[df["split"] == "test"].copy()

# Since we only have MegaDetector v6, we'll show the metrics comparison between CIS and TRANS
models = ["megadetectorv6"]
labels = ["MegaDetector v6"]

metrics = [("precision", "Precision"),
           ("recall", "Recall"), 
           ("F1", "F1 Score"),
           ("mAP50", "mAP@50"),
           ("mAP50-95", "mAP@50-95")]

# Option 1: Single model comparison (CIS vs TRANS)
def create_detector_comparison_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get CIS and TRANS data for the single model
    cis_data = test_df[test_df["domain"] == "cis"].iloc[0]
    trans_data = test_df[test_df["domain"] == "trans"].iloc[0]
    
    # Extract metric values
    cis_values = [cis_data[metric[0]] for metric in metrics]
    trans_values = [trans_data[metric[0]] for metric in metrics]
    metric_names = [metric[1] for metric in metrics]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    # Create bars
    bars_cis = ax.bar(x - width/2, cis_values, width, 
                     label='CIS Domain', color='#2E86AB', alpha=0.8)
    bars_trans = ax.bar(x + width/2, trans_values, width,
                       label='TRANS Domain', color='#F18F01', alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars_cis, cis_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    for bar, val in zip(bars_trans, trans_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Customize the plot
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('MegaDetector v6: CIS vs TRANS Domain Performance',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=0)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Option 2: Horizontal bar chart (better for metric names)
def create_detector_horizontal_plot():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get data
    cis_data = test_df[test_df["domain"] == "cis"].iloc[0]
    trans_data = test_df[test_df["domain"] == "trans"].iloc[0]
    
    cis_values = [cis_data[metric[0]] for metric in metrics]
    trans_values = [trans_data[metric[0]] for metric in metrics]
    metric_names = [metric[1] for metric in metrics]
    
    y = np.arange(len(metric_names))
    height = 0.35
    
    # Create horizontal bars
    bars_cis = ax.barh(y - height/2, cis_values, height,
                      label='CIS Domain', color='#2E86AB', alpha=0.8)
    bars_trans = ax.barh(y + height/2, trans_values, height,
                        label='TRANS Domain', color='#F18F01', alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars_cis, cis_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', ha='left', va='center', fontsize=10)
    
    for bar, val in zip(bars_trans, trans_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', ha='left', va='center', fontsize=10)
    
    # Customize
    ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('MegaDetector v6: CIS vs TRANS Domain Performance',
                 fontsize=14, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(metric_names)
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

# Option 3: If you want to extend to multiple models in the future
def create_extensible_detector_plot():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # This version can easily handle multiple models when you add them
    unique_models = test_df["model"].unique()
    model_labels = {"megadetectorv6": "MegaDetector v6"}  # Add more as needed
    
    x = np.arange(len(unique_models))
    width = 0.15
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A2B']
    
    for i, (metric_key, metric_name) in enumerate(metrics):
        cis_vals = []
        trans_vals = []
        
        for model in unique_models:
            cis_val = test_df[(test_df["model"] == model) & (test_df["domain"] == "cis")][metric_key].iloc[0]
            trans_val = test_df[(test_df["model"] == model) & (test_df["domain"] == "trans")][metric_key].iloc[0]
            cis_vals.append(cis_val)
            trans_vals.append(trans_val)
        
        offset = (i - 2) * width
        
        bars_cis = ax.bar(x + offset - width/2, cis_vals, width*0.8,
                         color=colors[i], alpha=0.8, 
                         label=f'{metric_name} (CIS)' if i == 0 else '')
        bars_trans = ax.bar(x + offset + width/2, trans_vals, width*0.8,
                           color=colors[i], alpha=0.6,
                           label=f'{metric_name} (TRANS)' if i == 0 else '')
        
        # Add value labels
        for bar, val in zip(bars_cis, cis_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        for bar, val in zip(bars_trans, trans_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Create custom legend
    legend_elements = []
    for i, (_, metric_name) in enumerate(metrics):
        legend_elements.append(plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.8,
                                           label=f'{metric_name} (CIS)'))
        legend_elements.append(plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.6,
                                           label=f'{metric_name} (TRANS)'))
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Detector Stage Performance: CIS vs TRANS Domain Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels.get(m, m) for m in unique_models])
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

# Call the function you prefer:
# Option 1: Simple comparison (recommended for single model)
create_detector_comparison_plot()

# Option 2: Horizontal bars (good for readability)
#create_detector_horizontal_plot()

# Option 3: Extensible for future models
#create_extensible_detector_plot()