#AI utilized
#https://github.com/wasserth/TotalSegmentator

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import argparse
from totalsegmentator.python_api import totalsegmentator

def calculate_metrics(pred, target):
    """Calculate Dice and Jaccard metrics."""
    # Ensure boolean arrays
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # Calculate intersection and union
    intersection = np.sum(pred & target)
    union = np.sum(pred | target)
    sum_both = np.sum(pred) + np.sum(target)
    
    # Calculate metrics
    dice = 2 * intersection / sum_both if sum_both > 0 else 0
    jaccard = intersection / union if union > 0 else 0
    
    return {'dice': dice, 'jaccard': jaccard}

def analyze_segmentations(base_folder, num_samples=15, fast=True):
    # Initialize results dictionary
    results = {
        'generated_vs_target': {'dice': {}, 'jaccard': {}},
        'input_vs_target': {'dice': {}, 'jaccard': {}}
    }
    
    # Process first sample to get structure names
    print("Running TotalSegmentator on first sample to identify structures...")
    target_path = os.path.join(base_folder, f'testimgTarget-1.nii')
    out_target = os.path.join(base_folder, f'seg_target_1')


    if not os.path.exists(out_target):
        totalsegmentator(target_path, out_target, fast=fast)
    
    # Get list of structures
    seg_files = [f for f in os.listdir(out_target) if f.endswith('.nii.gz')]
    
    print("\nProcessing all samples...")
    for i in range(1, num_samples + 1):
        print("Processing sample: ", i)
        # Paths for the current sample
        generated_path = os.path.join(base_folder, f'testimgGenerated-{i}.nii')
        input_path = os.path.join(base_folder, f'testimgInput-{i}.nii')
        target_path = os.path.join(base_folder, f'testimgTarget-{i}.nii')
        
        # Run TotalSegmentator
        out_generated = os.path.join(base_folder, f'seg_generated_{i}')
        out_input = os.path.join(base_folder, f'seg_input_{i}')
        out_target = os.path.join(base_folder, f'seg_target_{i}')
        
        if i > 1 and not os.path.exists(out_target):
            totalsegmentator(target_path, out_target, fast=fast)
        if not os.path.exists(out_generated):
            totalsegmentator(generated_path, out_generated, fast=fast)
        if not os.path.exists(out_input):
            totalsegmentator(input_path, out_input, fast=fast)
        
        # Compare each structure
        for seg_file in seg_files:
            structure_name = seg_file.replace('.nii.gz', '')
            
            # Load segmentations
            gen_seg = nib.load(os.path.join(out_generated, seg_file)).get_fdata() > 0
            input_seg = nib.load(os.path.join(out_input, seg_file)).get_fdata() > 0
            target_seg = nib.load(os.path.join(out_target, seg_file)).get_fdata() > 0

            if np.sum(target_seg) == 0:
                continue  # Skip structures that are empty in the target
            
            # Initialize structure in results if not present
            for comparison in results.values():
                for metric_dict in comparison.values():
                    if structure_name not in metric_dict:
                        metric_dict[structure_name] = []
            
            # Calculate metrics for generated vs target
            gen_metrics = calculate_metrics(gen_seg, target_seg)
            for metric, value in gen_metrics.items():
                results['generated_vs_target'][metric][structure_name].append(value)
            
            # Calculate metrics for input vs target
            input_metrics = calculate_metrics(input_seg, target_seg)
            for metric, value in input_metrics.items():
                results['input_vs_target'][metric][structure_name].append(value)
    
    return results

def save_results_to_file(results, base_folder):
    """Save results to text file."""
    results_file = os.path.join(base_folder, 'segmentation_results.txt')
    
    with open(results_file, 'w') as f:
        f.write("Results Summary\n")
        f.write("="*80 + "\n")
        
        # Get all structure names
        def get_mean_dice(structure, comparison_type):
            return np.mean(results[comparison_type]['dice'][structure])
        
        # Write Generated vs Target results
        f.write("\nGenerated vs Target (Sorted by mean Dice score)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Structure':<30} {'Dice (mean±std)':<20} {'Jaccard (mean±std)':<20}\n")
        f.write("-"*80 + "\n")
        
        # Sort structures by Generated vs Target Dice scores
        structures_gen = sorted(
            results['generated_vs_target']['dice'].keys(),
            key=lambda x: get_mean_dice(x, 'generated_vs_target'),
            reverse=True
        )
        
        for structure in structures_gen:
            gen_dice = np.array(results['generated_vs_target']['dice'][structure])
            gen_jacc = np.array(results['generated_vs_target']['jaccard'][structure])
            
            dice_stat = f"{gen_dice.mean():.3f}±{gen_dice.std():.3f}"
            jacc_stat = f"{gen_jacc.mean():.3f}±{gen_jacc.std():.3f}"
            
            f.write(f"{structure:<30} {dice_stat:<20} {jacc_stat:<20}\n")
        
        # Write Input vs Target results
        f.write("\nInput vs Target (Sorted by mean Dice score)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Structure':<30} {'Dice (mean±std)':<20} {'Jaccard (mean±std)':<20}\n")
        f.write("-"*80 + "\n")
        
        # Sort structures by Input vs Target Dice scores
        structures_input = sorted(
            results['input_vs_target']['dice'].keys(),
            key=lambda x: get_mean_dice(x, 'input_vs_target'),
            reverse=True
        )
        
        for structure in structures_input:
            input_dice = np.array(results['input_vs_target']['dice'][structure])
            input_jacc = np.array(results['input_vs_target']['jaccard'][structure])
            
            dice_stat = f"{input_dice.mean():.3f}±{input_dice.std():.3f}"
            jacc_stat = f"{input_jacc.mean():.3f}±{input_jacc.std():.3f}"
            
            f.write(f"{structure:<30} {dice_stat:<20} {jacc_stat:<20}\n")
    
    print(f"Results saved to: {results_file}")
    return results_file

def print_summary(results):
    """Print summary statistics sorted by mean Dice score."""
    print("\nResults Summary")
    print("="*80)
    
    # Get all structure names and sort by mean Dice score
    def get_mean_dice(structure, comparison_type):
        return np.mean(results[comparison_type]['dice'][structure])
    
    # Print Generated vs Target results
    print("\nGenerated vs Target (Sorted by mean Dice score)")
    print("-"*80)
    print(f"{'Structure':<30} {'Dice (mean±std)':<20} {'Jaccard (mean±std)':<20}")
    print("-"*80)
    
    # Sort structures by Generated vs Target Dice scores
    structures_gen = sorted(
        results['generated_vs_target']['dice'].keys(),
        key=lambda x: get_mean_dice(x, 'generated_vs_target'),
        reverse=True
    )
    
    for structure in structures_gen:
        gen_dice = np.array(results['generated_vs_target']['dice'][structure])
        gen_jacc = np.array(results['generated_vs_target']['jaccard'][structure])
        
        dice_stat = f"{gen_dice.mean():.3f}±{gen_dice.std():.3f}"
        jacc_stat = f"{gen_jacc.mean():.3f}±{gen_jacc.std():.3f}"
        
        print(f"{structure:<30} {dice_stat:<20} {jacc_stat:<20}")
    
    # Print Input vs Target results
    print("\nInput vs Target (Sorted by mean Dice score)")
    print("-"*80)
    print(f"{'Structure':<30} {'Dice (mean±std)':<20} {'Jaccard (mean±std)':<20}")
    print("-"*80)
    
    # Sort structures by Input vs Target Dice scores
    structures_input = sorted(
        results['input_vs_target']['dice'].keys(),
        key=lambda x: get_mean_dice(x, 'input_vs_target'),
        reverse=True
    )
    
    for structure in structures_input:
        input_dice = np.array(results['input_vs_target']['dice'][structure])
        input_jacc = np.array(results['input_vs_target']['jaccard'][structure])
        
        dice_stat = f"{input_dice.mean():.3f}±{input_dice.std():.3f}"
        jacc_stat = f"{input_jacc.mean():.3f}±{input_jacc.std():.3f}"
        
        print(f"{structure:<30} {dice_stat:<20} {jacc_stat:<20}")


def visualize_results(results, top_n=3):
    """
    Create bar plots comparing Dice and Jaccard scores for all structures.
    
    Parameters:
    results: Dictionary containing the metrics
    top_n: Optional, number of top structures to show (by Dice score)
    """
    # Get structures and sort by generated Dice score
    structures = sorted(results['input_vs_target']['dice'].keys(), key=lambda s: np.mean(results['input_vs_target']['dice'][s]), reverse=True)
    
    # Limit to top_n if specified
    if top_n is not None:
        structures = structures[:top_n]
    
    # Prepare data
    gen_dice_means = [np.mean(results['generated_vs_target']['dice'][s]) for s in structures]
    gen_dice_stds = [np.std(results['generated_vs_target']['dice'][s]) for s in structures]
    input_dice_means = [np.mean(results['input_vs_target']['dice'][s]) for s in structures]
    input_dice_stds = [np.std(results['input_vs_target']['dice'][s]) for s in structures]
    
    gen_jacc_means = [np.mean(results['generated_vs_target']['jaccard'][s]) for s in structures]
    gen_jacc_stds = [np.std(results['generated_vs_target']['jaccard'][s]) for s in structures]
    input_jacc_means = [np.mean(results['input_vs_target']['jaccard'][s]) for s in structures]
    input_jacc_stds = [np.std(results['input_vs_target']['jaccard'][s]) for s in structures]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Set width of bars and positions of bars
    bar_width = 0.35
    x = np.arange(len(structures))
    
    # Plot Dice scores
    ax1.bar(x - bar_width/2, gen_dice_means, bar_width, 
            label='Generated vs Target', color='cornflowerblue',
            yerr=gen_dice_stds, capsize=5)
    ax1.bar(x + bar_width/2, input_dice_means, bar_width,
            label='Input vs Target', color='lightcoral',
            yerr=input_dice_stds, capsize=5)
    
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Dice Scores by Structure')
    ax1.set_xticks(x)
    ax1.set_xticklabels(structures, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.2)
    
    # Plot Jaccard scores
    ax2.bar(x - bar_width/2, gen_jacc_means, bar_width,
            label='Generated vs Target', color='cornflowerblue',
            yerr=gen_jacc_stds, capsize=5)
    ax2.bar(x + bar_width/2, input_jacc_means, bar_width,
            label='Input vs Target', color='lightcoral',
            yerr=input_jacc_stds, capsize=5)
    
    ax2.set_ylabel('Jaccard Index')
    ax2.set_title('Jaccard Indices by Structure')
    ax2.set_xticks(x)
    ax2.set_xticklabels(structures, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.2)
    
    # Adjust layout and display
    plt.tight_layout()
    
    return fig



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default="")

    args = parser.parse_args()

    base_folder = args.folder
    results = analyze_segmentations(base_folder, 213, fast=False)
    save_results_to_file(results, base_folder)
    print_summary(results)
    fig = visualize_results(results, 20)
    plt.savefig(base_folder + '/segmentation_results.png', dpi=300, bbox_inches='tight')
    plt.close()