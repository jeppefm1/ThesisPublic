import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# W&B settings
ENTITY = ""
PROJECT = "medDDPM"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

# Metrics to keep
selected_columns = ["step", "validation_loss", "training_loss", "SSIM", "PSNR", "val_noise_pred_loss", "time_elapsed (min)"]

all_metrics = []

for run in runs:
    history = run.history(samples=100000)  # Fetch all data
    
    if history.empty:
        print(f"⚠️ No history found for run {run.name} ({run.id}). Skipping...")
        continue

    # Keep only selected columns and ensure all exist
    history = history[selected_columns] if all(col in history.columns for col in selected_columns) else history.reindex(columns=selected_columns)

    # Handle NaN or infinite values before computing the epoch
    history = history.dropna(subset=["step"])  # Drop rows where 'step' is NaN
    history = history[history["step"].apply(pd.to_numeric, errors='coerce').notnull()]  # Remove non-numeric 'step'

    # Compute epoch (ensure integer division)
    history["epoch"] = (history["step"] // 120).astype(int)

     
    if "time_elapsed (min)" in history.columns and not history["time_elapsed (min)"].empty:
        # First, forward-fill any missing values to handle gaps
        filled_time = history["time_elapsed (min)"].ffill()
        
        cumulative_time = []
        total = 0
        prev_value = 0

        for current_value in filled_time:
            if pd.isna(current_value):
                cumulative_time.append(float('nan'))  # Keep NaN values as NaN
                continue
                
            if current_value >= prev_value:  # Time is increasing or stable
                cumulative_time.append(total + current_value)  # Add current time to the total
                prev_value = current_value  # Update prev_value to the current value
            else:  # Time has reset
                total += prev_value  # Add the last valid time to the total before reset
                cumulative_time.append(total + current_value)  # Add current time after reset
                prev_value = current_value  # Set prev_value to the current value (reset point)
        
        # Make sure the length matches
        if len(cumulative_time) == len(history):
            history["cumulative_time (hours)"] = [t/60 if not pd.isna(t) else t for t in cumulative_time]
        else:
            # Handle length mismatch by padding with NaN
            padded_time = cumulative_time + [float('nan')] * (len(history) - len(cumulative_time))
            history["cumulative_time (hours)"] = [t/60 if not pd.isna(t) else t for t in padded_time[:len(history)]]
    else:
        # Create column with all NaN if time data is missing
        history["cumulative_time (hours)"] = float('nan')


    # Add run metadata
    history["run_id"] = run.id
    history["run_name"] = run.name
    all_metrics.append(history)

if all_metrics:
    # Concatenate all runs
    df = pd.concat(all_metrics, ignore_index=True)

    # Aggregate by epoch (mean of losses and other metrics for each epoch)
    df_agg = df.groupby(["run_name", "epoch"]).agg({
        "validation_loss": "mean",
        "training_loss": "mean",
        "SSIM": "mean",
        "PSNR": "mean",
        "val_noise_pred_loss": "mean",
        "cumulative_time (hours)": "max"
    }).reset_index()

    # Save aggregated data
    df_agg.to_csv("wandb_epoch_metrics.csv", index=False)
    print("✅ Metrics saved successfully!")

        # 🎨 Create subplots for each run
    metrics_to_plot = [ "training_loss", "val_noise_pred_loss", "SSIM", "PSNR", "validation_loss", "cumulative_time (hours)"]
    
    # Create a plot for each run
    for run_name in df_agg["run_name"].unique():
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))  # 3 rows, 2 columns of subplots
        axes = axes.flatten()  # Flatten axes to index easily
        
        # Filter the data for the current run
        run_data = df_agg[df_agg["run_name"] == run_name]

        for i, metric in enumerate(metrics_to_plot):
            if metric in run_data.columns:
               # Filter out non-positive values (log scale requires y > 0)
                plot_data = run_data[run_data[metric] > 0]

                if not plot_data.empty:
                    
                    if metric != "cumulative_time (hours)": 
                        sns.lineplot(data=plot_data, x="epoch", y=metric, ax=axes[i], marker="o", markersize=4, alpha=1, color="darkblue")
                        axes[i].set_yscale("log")  # Set log scale
                        if metric == "validation_loss":  
                            axes[i].set_title("validation loss on image generation (log scale)") 
                        elif metric == "val_noise_pred_loss":  
                            axes[i].set_title("validation loss on noise prediction (log scale)") 
                        else:
                            axes[i].set_title(f"{metric.replace('_', ' ')} (log scale)") 
                    
                    else: 
                        sns.lineplot(data=plot_data, x="epoch", y=metric, ax=axes[i], marker="o", markersize=3, alpha=0.8, color="darkblue")
                        axes[i].set_title(f"{metric.replace('_', ' ')}")
                    axes[i].set_xlabel("Epoch")
                    axes[i].set_ylabel(metric.replace("_", " "))
                    axes[i].grid()
                else:
                    axes[i].set_visible(False)  # Hide the subplot if there's nothing to show
        
        plt.tight_layout()
        plt.savefig(f"medDDPMHendrix/{run_name}_metrics.png")
        plt.close()
        print(f"✅ Saved plot for run {run_name}.")

else:
    print("❌ No valid runs found with the requested metrics.")
