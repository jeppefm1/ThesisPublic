import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# W&B settings
ENTITY = ""
PROJECT = "Vox2Vox"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

# Metrics to keep
selected_columns = ["epoch", "epoch_gan_loss", "epoch_voxel_loss", "epoch_loss_G","epoch_loss_D","validation_loss","epoch_d_accuracy", "SSIM", "PSNR"]

all_metrics = []

for run in runs:
    history = run.history(samples=100000)  # Fetch all data
    
    if history.empty:
        print(f"⚠️ No history found for run {run.name} ({run.id}). Skipping...")
        continue

    # Keep only selected columns and ensure all exist
    history = history[selected_columns] if all(col in history.columns for col in selected_columns) else history.reindex(columns=selected_columns)

    # Add run metadata
    history["run_id"] = run.id
    history["run_name"] = run.name
    all_metrics.append(history)

if all_metrics:
    # Concatenate all runs
    df = pd.concat(all_metrics, ignore_index=True)

        # 🎨 Create subplots for each run
    metrics_to_plot = ["epoch_gan_loss", "epoch_voxel_loss", "epoch_loss_G","epoch_loss_D","validation_loss","epoch_d_accuracy", "SSIM", "PSNR"]
    
    # Create a plot for each run
    for run_name in df["run_name"].unique():
        fig, axes = plt.subplots(4, 2, figsize=(15, 15))
        axes = axes.flatten()  # Flatten axes to index easily
        
        # Filter the data for the current run
        run_data = df[df["run_name"] == run_name]

        for i, metric in enumerate(metrics_to_plot):
            if metric in run_data.columns:
                # Filter out non-positive values (log scale requires y > 0)
                plot_data = run_data[run_data[metric] > 0]

                if not plot_data.empty:
                    sns.lineplot(data=plot_data, x="epoch", y=metric, ax=axes[i], marker="o", markersize=4,  alpha=0.7, color="darkblue")
                    if metric != "cumulative_time (hours)": axes[i].set_yscale("log")  # Set log scale
                    if metric != "cumulative_time (hours)": axes[i].set_title(f"{metric.replace('_', ' ')} (log scale)") 
                    else: axes[i].set_title(f"{metric.replace('_', ' ')}")
                    axes[i].set_xlabel("Epoch")
                    axes[i].set_ylabel(metric.replace("_", " "))
                    axes[i].grid()
                else:
                    axes[i].set_visible(False)  # Hide the subplot if there's nothing to show
        
        plt.tight_layout()
        plt.savefig(f"Vox2VoxHendrix/{run_name}_metrics.png")
        plt.close()
        print(f"✅ Saved plot for run {run_name}.")

else:
    print("❌ No valid runs found with the requested metrics.")
