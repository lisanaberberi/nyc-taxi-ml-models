import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# Load the CSV
metrics_df = pd.read_csv("results/metrics_custom_baseline.csv")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(style="whitegrid")

# Define colors
color_baseline = '#A0C4FF'  
color_custom = '#BDB2FF' 

# Determine model type from model_name
metrics_df['model_type'] = metrics_df['model_name'].apply(
    lambda x: 'baseline' if x.startswith('baseline_') else 'custom'
)

# Create a color list based on model type
colors = metrics_df['model_type'].map({
    'baseline': color_baseline,
    'custom': color_custom
})

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# Bar positions
index = np.arange(len(metrics_df))

# R² bars with conditional colors
r2_bars = ax.bar(index, metrics_df['r2'], width=0.4, color=colors)

# Add value labels on top of bars
for i, (bar, value) in enumerate(zip(r2_bars, metrics_df['r2'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.3f}', ha='center', va='bottom', fontsize=9)

# Labels and ticks
ax.set_xlabel('Model Name', fontsize=10)
ax.set_ylabel('R² Score', fontsize=10)
ax.tick_params(axis='y')
ax.set_xticks(index)
ax.set_xticklabels(metrics_df['model_name'], rotation=45, ha='right')
ax.set_ylim(0, 1)

# Y-axis ticks every 0.10
ax.set_yticks(np.arange(0, 1.05, 0.10))

# Title and grid
plt.title("R² Score per Experiment-Run", fontsize=16, pad=30)
ax.grid(axis='y', alpha=0.3)

# Custom legend
legend_elements = [
    Patch(facecolor=color_baseline, label='Baseline Model'),
    Patch(facecolor=color_custom, label='Custom Model')
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2)

plt.tight_layout()
plt.savefig("results/r2_metrics_plot_colored.png", dpi=300, bbox_inches='tight')
plt.show()

# Print model type assignments for verification
print("\nModel Type Assignments:")
for idx, row in metrics_df.iterrows():
    print(f"{row['model_name']}: {row['model_type']} (R² = {row['r2']:.3f})")

# Print summary statistics
baseline_models = metrics_df[metrics_df['model_type'] == 'baseline']
custom_models = metrics_df[metrics_df['model_type'] == 'custom']

print(f"\nSummary Statistics:")
print(f"Baseline Models - Mean R²: {baseline_models['r2'].mean():.3f}, Count: {len(baseline_models)}")
print(f"Custom Models - Mean R²: {custom_models['r2'].mean():.3f}, Count: {len(custom_models)}")

if len(baseline_models) > 0 and len(custom_models) > 0:
    improvement = ((custom_models['r2'].mean() - baseline_models['r2'].mean()) / baseline_models['r2'].mean()) * 100
    print(f"Average R² improvement: {improvement:+.1f}%")