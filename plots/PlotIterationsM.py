import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Base path for reconstructions
base_path = "jeppes_project/data/reconstructionsDiffIteerOSart"

# Function to extract metrics from a text file
def read_metrics_file(file_path):
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Split by colon and strip whitespace
                parts = line.strip().split(': ')
                # Handle single-line time format (OSSART75 case)
                if len(parts) == 1 and line.strip().startswith('Time:'):
                    parts = ['Time', line.strip().split('Time:')[1].strip()]
                
                if len(parts) == 2:
                    metric_name, value = parts
                    # Clean up metric name
                    metric_name = metric_name.strip()
                    
                    # Try to convert to float
                    try:
                        # Handle the time metric: convert seconds to minutes
                        if metric_name.lower() == 'time':
                            metrics['reconstruction_time_sec'] = float(value)
                            metrics['reconstruction_time_min'] = float(value) / 60.0
                        else:
                            metrics[metric_name] = float(value)
                    except ValueError:
                        metrics[metric_name] = value
        return metrics
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Function to extract iteration count from folder name
def extract_iteration(folder_name):
    match = re.search(r'OSSART(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None

# Collect all metric data
def collect_metrics():
    data = []
    
    # Get all OSSART folders
    OSSART_folders = [f for f in os.listdir(base_path) if f.startswith('OSSART')]
    
    for OSSART_folder in OSSART_folders:
        iterations = extract_iteration(OSSART_folder)
        if iterations is None:
            continue
            
        OSSART_path = os.path.join(base_path, OSSART_folder)
        scan_types = [d for d in os.listdir(OSSART_path) if os.path.isdir(os.path.join(OSSART_path, d))]
        
        for scan_type in scan_types:
            scan_type_path = os.path.join(OSSART_path, scan_type)
            metric_files = [f for f in os.listdir(scan_type_path) if f.endswith('_metrics.txt')]
            
            for metric_file in metric_files:
                file_path = os.path.join(scan_type_path, metric_file)
                metrics = read_metrics_file(file_path)
                
                if metrics:
                    scan_id = metric_file.replace('_metrics.txt', '')
                    entry = {
                        'iterations': iterations,
                        'scan_type': scan_type,
                        'scan_id': scan_id,
                        **metrics
                    }
                    data.append(entry)
    
    # Add reference with perfect scores
    # For each scan type that appears in the data
    if data:
        df = pd.DataFrame(data)
        for scan_type in df['scan_type'].unique():
            # Get all unique scan IDs for this scan type
            scan_ids = df[df['scan_type'] == scan_type]['scan_id'].unique()
            
            for scan_id in scan_ids:
                # Add an entry for method with perfect scores
                perfect_entry = {
                    'iterations': 75,
                    'scan_type': scan_type,
                    'scan_id': scan_id,
                    'MAE': 0.0,
                    'RMSE': 0.0,
                    'SSIM': 1.0,
                    'PSNR': float('inf')  # Technically infinity, but we'll handle this special case
                }
                
                # If we have reconstruction time data for other iterations of this scan type/id,
                # use the actual time data if available, otherwise estimate
                if 'reconstruction_time_sec' not in df[(df['scan_type'] == scan_type) & 
                                               (df['scan_id'] == scan_id) & 
                                               (df['iterations'] == 75)].columns:
                    # If no actual time data , estimate it
                    scan_times = df[(df['scan_type'] == scan_type) & 
                                   (df['scan_id'] == scan_id) & 
                                   (df['iterations'] != 75)]
                    
                    if not scan_times.empty and 'reconstruction_time_sec' in scan_times.columns:
                        # Simple linear projection based on average time per iteration
                        avg_time_per_iter = scan_times['reconstruction_time_sec'].mean() / scan_times['iterations'].mean()
                        perfect_entry['reconstruction_time_sec'] = avg_time_per_iter * 75
                        perfect_entry['reconstruction_time_min'] = perfect_entry['reconstruction_time_sec'] / 60.0
                
                data.append(perfect_entry)
    
    return pd.DataFrame(data)

# Create plots
def create_plots(df):
    # Handle infinity values for plotting
    df.replace([float('inf'), -float('inf')], np.nan, inplace=True)
    
    # For PSNR, use a very high value instead of infinity
    max_psnr = df['PSNR'].max()
    if not np.isnan(max_psnr):
        # Use 20% higher than the max non-infinity value
        high_psnr = max_psnr * 1.2
        df.loc[df['iterations'] == 75, 'PSNR'] = high_psnr
    
    # List of metrics to plot
    metrics = ['MAE', 'RMSE', 'SSIM', 'PSNR']
    
    # Check if reconstruction time exists in the dataframe
    time_metric = None
    if 'reconstruction_time_min' in df.columns:
        time_metric = 'reconstruction_time_min'
        metrics.append(time_metric)
    elif 'reconstruction_time_sec' in df.columns:
        time_metric = 'reconstruction_time_sec'
        metrics.append(time_metric)
    
    # Get unique scan types and iterations
    scan_types = df['scan_type'].unique()
    iterations = sorted(df['iterations'].unique())
    
    # Colors for different scan types
    colors = plt.cm.tab10(np.linspace(0, 1, len(scan_types)))
    
    # Create plots
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for i, scan_type in enumerate(scan_types):
            # Filter data for this scan type
            scan_data = df[df['scan_type'] == scan_type]
            
            # Skip if this scan type doesn't have this metric
            if metric not in scan_data.columns or scan_data[metric].isnull().all():
                continue
            
            # Group by iterations and calculate mean and std
            grouped = scan_data.groupby('iterations')[metric].agg(['mean', 'std']).reset_index()
            
            # Special handling for reference point for quality metrics
            if 75 in grouped['iterations'].values and metric not in [time_metric]:
                ref_point = grouped[grouped['iterations'] == 75]
                regular_points = grouped[grouped['iterations'] != 75]
                
                # Plot regular points
                plt.errorbar(
                    regular_points['iterations'], 
                    regular_points['mean'], 
                    yerr=regular_points['std'],
                    fmt='o-',
                    color=colors[i],
                    capsize=5,                 # Adds horizontal bars at the end of error bars
                    elinewidth=1.5,            # Make error bars thicker
                    capthick=2,                # Make cap bars prominent
                    label=f'{scan_type}'
                )
                # Add a filled region for std deviation with transparency
                plt.fill_between(
                    grouped['iterations'], 
                    grouped['mean'] - grouped['std'], 
                    grouped['mean'] + grouped['std'], 
                    color=colors[i], 
                    alpha=0.1  # Transparency for better visibility of overlaps
)
                
                # Plot reference point with star marker
                plt.errorbar(
                    ref_point['iterations'], 
                    ref_point['mean'], 
                    yerr=ref_point['std'],
                    fmt='*',
                    markersize=10,
                    color=colors[i]
                )
                
                # Add a text label for the reference point
                for _, row in ref_point.iterrows():
                    plt.annotate(
                        f"Reference\n(OSSART75)", 
                        (row['iterations'], row['mean']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center'
                    )
            else:
                # Plot all points normally if no reference point
                plt.errorbar(
                    grouped['iterations'], 
                    grouped['mean'], 
                    yerr=grouped['std'],
                    fmt='o-',
                    color=colors[i],
                    capsize=5,                 # Adds horizontal bars at the end of error bars
                    elinewidth=1.5,            # Make error bars thicker
                    capthick=2,                # Make cap bars prominent
                    label=f'{scan_type}'
                )
                # Add a filled region for std deviation with transparency
                plt.fill_between(
                    grouped['iterations'], 
                    grouped['mean'] - grouped['std'], 
                    grouped['mean'] + grouped['std'], 
                    color=colors[i], 
                    alpha=0.1  # Transparency for better visibility of overlaps
)
        
        # Determine title suffix and labels based on metric
        title_suffix = ""
        y_label = metric
        
        if metric in ['SSIM', 'PSNR']:
            title_suffix = " (Higher is Better)"
        elif metric in ['MAE', 'RMSE']:
            title_suffix = " (Lower is Better)"
        elif metric == 'reconstruction_time_min':
            title_suffix = " (Lower is Better)"
            y_label = "Reconstruction Time (minutes)"
        elif metric == 'reconstruction_time_sec':
            title_suffix = " (Lower is Better)"
            y_label = "Reconstruction Time (seconds)"
            
        # Set plot title and labels
        plt.title(f'{y_label} vs. Number of OSSART Iterations{title_suffix}')
        plt.xlabel('Number of Iterations')
        plt.ylabel(y_label)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Set x-ticks to include all iterations
        plt.xticks(iterations)
        
        # Determine if y-axis should start from zero
        if metric in ['MAE', 'RMSE', 'reconstruction_time_min', 'reconstruction_time_sec'] and plt.ylim()[0] > 0:
            plt.ylim(bottom=0)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = 'metrics_plots'
        os.makedirs(output_dir, exist_ok=True)
        
        # Use appropriate filename based on metric
        filename = metric
        if metric == 'reconstruction_time_min':
            filename = 'reconstruction_time'
        elif metric == 'reconstruction_time_sec':
            filename = 'reconstruction_time_seconds'
            
        plt.savefig(os.path.join(output_dir, f'{filename}_vs_iterations.png'), dpi=300)
        plt.close()
    
    # Create an additional plot comparing reconstruction time with quality metrics
    if time_metric:
        for metric in ['MAE', 'RMSE', 'SSIM', 'PSNR']:
            if metric not in df.columns:
                continue
                
            plt.figure(figsize=(10, 6))
            
            for i, scan_type in enumerate(scan_types):
                # Filter data for this scan type and check if we have both metrics
                scan_data = df[df['scan_type'] == scan_type]
                
                if metric not in scan_data.columns or time_metric not in scan_data.columns:
                    continue
                    
                # Remove any reference points (iteration=75) for this plot
                scan_data = scan_data[scan_data['iterations'] != 75]
                
                if not scan_data.empty:
                    plt.scatter(
                        scan_data[time_metric][0::5],
                        scan_data[metric][0::5],
                        color=colors[i],
                        label=f'{scan_type}',
                        alpha=0.7
                    )
                    
                    # Add iteration labels to each point
                    for _, row in scan_data[0::5].iterrows():
                        plt.annotate(
                            f"{int(row['iterations'])}",
                            (row[time_metric], row[metric]),
                            textcoords="offset points",
                            xytext=(0, 5),
                            ha='center',
                            fontsize=8
                        )
            
            # Determine title suffix based on metric
            title_suffix = ""
            if metric in ['SSIM', 'PSNR']:
                title_suffix = " (Higher is Better)"
            elif metric in ['MAE', 'RMSE']:
                title_suffix = " (Lower is Better)"
            
            # Set appropriate time label
            time_label = "Reconstruction Time (minutes)" if time_metric == 'reconstruction_time_min' else "Reconstruction Time (seconds)"
                
            plt.title(f'{metric} vs. {time_label}{title_suffix}')
            plt.xlabel(time_label)
            plt.ylabel(metric)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.tight_layout()
            
            # Save the plot
            time_suffix = "minutes" if time_metric == 'reconstruction_time_min' else "seconds"
            plt.savefig(os.path.join(output_dir, f'{metric}_vs_reconstruction_time_{time_suffix}.png'), dpi=300)
            plt.close()

# Main function
def main():
    print("Collecting metrics data...")
    df = collect_metrics()
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"Found {len(df)} metric entries across {df['scan_type'].nunique()} scan types.")
    
    # Create plots
    print("Creating plots...")
    create_plots(df)
    
    print(f"Plots saved to 'metrics_plots' directory.")
    
    # Print summary statistics
    print("\nSummary Statistics by Scan Type and Iteration:")
    metrics_to_report = ['MAE', 'RMSE', 'SSIM', 'PSNR']
    
    # Add time metrics to report
    if 'reconstruction_time_min' in df.columns:
        metrics_to_report.append('reconstruction_time_min')
    elif 'reconstruction_time_sec' in df.columns:
        metrics_to_report.append('reconstruction_time_sec')
        
    for scan_type in df['scan_type'].unique():
        print(f"\n{scan_type}:")
        summary_metrics = {metric: ['mean', 'std'] for metric in metrics_to_report if metric in df.columns}
        if summary_metrics:
            summary = df[df['scan_type'] == scan_type].groupby('iterations').agg(summary_metrics)
            print(summary)

if __name__ == "__main__":
    main()