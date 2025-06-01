import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_csv_files(base_dir):
    """Find all comparison summary CSV files in the base directory."""
    pattern = os.path.join(base_dir, '*_comparison_summary.csv')
    return glob.glob(pattern)

def create_dashboard(data_dir, output_dir=None):
    """Create a comprehensive dashboard with subplots for all metrics."""
    # Find all CSV files
    csv_files = find_csv_files(data_dir)
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Combine all CSV files
    all_data = []
    for file in csv_files:
        scan_name = os.path.basename(file).replace('_comparison_summary.csv', '')
        df = pd.read_csv(file)
        df['ScanName'] = scan_name
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Calculate percentage of projections for each scan
    # First identify the Full projection count for each unique scan
    scan_full_projs = {}
    for scan_name in combined_df['ScanName'].unique():
        scan_data = combined_df[combined_df['ScanName'] == scan_name]
        full_data = scan_data[scan_data['SamplingName'] == 'Full']
        if not full_data.empty:
            scan_full_projs[scan_name] = full_data['NumProj'].iloc[0]
    
    # Calculate projection percentage for each row
    def get_proj_percentage(row):
        if row['ScanName'] in scan_full_projs:
            return (row['NumProj'] / scan_full_projs[row['ScanName']]) * 100
        return np.nan
    
    combined_df['ProjectionPercentage'] = combined_df.apply(get_proj_percentage, axis=1)
    
    # Group by sampling name to get average percentage
    sampling_avg_percentage = combined_df.groupby('SamplingName')['ProjectionPercentage'].mean()
    
    # Define color mapping for sampling patterns
    sampling_colors = {
        'Full': 'blue',
        '33pct': 'green',
        '10pct': 'red'
    }
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metrics = ['MAE', 'RMSE', 'SSIM', 'PSNR']
    metric_titles = {
        'MAE': 'Mean Absolute Error',
        'RMSE': 'Root Mean Square Error',
        'SSIM': 'Structural Similarity Index Measure',
        'PSNR': 'Peak Signal-to-Noise Ratio (dB)'
    }
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Group by sampling pattern and plot for each
        for sampling, color in sampling_colors.items():
            sampling_data = combined_df[combined_df['SamplingName'] == sampling]
            if not sampling_data.empty:
                # Get average across all scans, grouped by iteration count
                avg_by_iter = sampling_data.groupby('Iterations')[metric].mean()
                std_by_iter = sampling_data.groupby('Iterations')[metric].std()
                
                # Get average percentage for this sampling pattern
                avg_percentage = sampling_avg_percentage.get(sampling, np.nan)
                percentage_label = f"{avg_percentage:.1f}%" if not np.isnan(avg_percentage) else "??%"
                
                ax.plot(
                    avg_by_iter.index,
                    avg_by_iter.values,
                    color=color,
                    marker='o',
                    markersize=8,
                    linewidth=2,
                    label=f"{sampling} ({percentage_label} of projections)"
                )
                ax.fill_between(
                    avg_by_iter.index, 
                    avg_by_iter - std_by_iter, 
                    avg_by_iter + std_by_iter, 
                    color=color, 
                    alpha=0.1 )
        
        # Set plot properties
        ax.set_title(f"Average {metric_titles[metric]} Across Scans", fontsize=14)
        ax.set_xlabel("Iterations", fontsize=12)
        ax.set_ylabel(metric_titles[metric], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust axis for SSIM
        if metric == 'SSIM':
            ax.set_ylim(0, 1)
    
    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.05))
    
    plt.suptitle("CT Reconstruction Quality Metrics vs Iterations for the OS-SART Algorithm", fontsize=18)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Save figure if output directory is provided
    if output_dir:
        plt.savefig(os.path.join(output_dir, "metrics_dashboard.png"), dpi=300)
        print(f"Saved dashboard to {output_dir}")
    
    plt.show()
    
    # Create time plot separately
    plt.figure(figsize=(10, 6))
    
    for sampling, color in sampling_colors.items():
        sampling_data = combined_df[combined_df['SamplingName'] == sampling]
        if not sampling_data.empty:
            # Aggregate by iteration count
            avg_by_iter = sampling_data.groupby('Iterations')['Time'].mean()
            std_by_iter = sampling_data.groupby('Iterations')['Time'].std()
            
            # Get average percentage for this sampling pattern
            avg_percentage = sampling_avg_percentage.get(sampling, np.nan)
            percentage_label = f"{avg_percentage:.1f}%" if not np.isnan(avg_percentage) else "??%"
            
            plt.plot(
                avg_by_iter.index,
                avg_by_iter.values,
                color=color,
                marker='o',
                markersize=8,
                linewidth=2,
                label=f"{sampling} ({percentage_label} of projections)"
            )

            plt.fill_between(
                    avg_by_iter.index, 
                    avg_by_iter - std_by_iter, 
                   avg_by_iter +std_by_iter, 
                    color=color, 
                    alpha=0.1 )
    
    plt.title("Average Reconstruction Time Across Scans vs Iterations for the OS-SART Algorithm", fontsize=16)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Time (seconds)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure if output directory is provided
    if output_dir:
        plt.savefig(os.path.join(output_dir, "time_vs_iterations.png"), dpi=300)
        print(f"Saved time plot to {output_dir}")
    
    plt.show()
    
    # Add additional summary statistics
    print("\nSummary Statistics:")
    print("-" * 40)
    print("Average projection percentages by sampling pattern:")
    for sampling, percentage in sampling_avg_percentage.items():
        print(f"  {sampling}: {percentage:.1f}%")
    
    print("\nAverage metrics at 75 iterations:")
    metrics_75 = combined_df[combined_df['Iterations'] == 75].groupby('SamplingName')[metrics].mean()
    print(metrics_75)
    
    print("\nAverage reconstruction time at 75 iterations:")
    time_75 = combined_df[combined_df['Iterations'] == 75].groupby('SamplingName')['Time'].mean()
    print(time_75)


if __name__ == "__main__":
    # Path to directory containing the CSV files
    data_directory = "/media/15tb_encrypted/jeppes_project/data/reconstructions"
    
    # Path to save the output plots (optional)
    output_directory = "/media/15tb_encrypted/jeppes_project/Thesis/plots/plots"
    
    # Create dashboard
    create_dashboard(data_directory, output_directory)