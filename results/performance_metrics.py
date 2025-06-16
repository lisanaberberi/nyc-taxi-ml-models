import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Load the new CSV
#metrics_df = pd.read_csv("results/metrics_results_baseline.csv")

#split between baseline and custom metrics
metrics_df = pd.read_csv("results/metrics_results_custom.csv")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(style="whitegrid")
color_mae = '#A0C4FF'
color_mse = '#BDB2FF'


# Create figure and axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar positions
bar_width = 0.4
index = np.arange(len(metrics_df))

# MAE bars (left y-axis)
mae_bars = ax1.bar(index - bar_width/2, metrics_df['mae'], bar_width, color=color_mae, label='MAE')
ax1.set_xlabel('Run Name', fontsize=12)
ax1.set_ylabel('MAE', fontsize=12, color=color_mae)
ax1.tick_params(axis='y', labelcolor=color_mae)
ax1.set_xticks(index)
ax1.set_xticklabels(metrics_df['model_name'], rotation=45, ha='right')


# MSE bars (right y-axis)
ax2 = ax1.twinx()
mse_bars = ax2.bar(index + bar_width/2, metrics_df['mse'], bar_width, color=color_mse, label='MSE')
ax2.set_ylabel('MSE', fontsize=12, color=color_mse)
ax2.tick_params(axis='y', labelcolor=color_mse)



# Title and grid
plt.title("MAE and MSE per Experiment-Run", fontsize=16, pad=30)
ax1.grid(axis='y')

# Lighter grid on y-axis
# ax1.grid(axis='y', color='lightgrey', linewidth=0.1, alpha=0.3)

# Legends — grab one bar from each and position legend at top center
lines = [mae_bars[0], mse_bars[0]]
labels = ['MAE', 'MSE']
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2)

plt.tight_layout()
#plt.savefig("results/baseline_metrics_dual_axis_plot.png")
plt.savefig("results/custom_metrics_dual_axis_plot.png")
plt.show()
