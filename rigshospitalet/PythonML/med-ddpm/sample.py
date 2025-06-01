#Based on https://github.com/mobaidoctor/med-ddpm
#AI utilized

from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import NIFTIDataset
from torchvision.transforms import Compose, Lambda
import nibabel as nib
import torchio as tio
import numpy as np
import argparse
import torch
import os
import glob
import torch.nn.functional as F
import skimage.metrics
from utils.toHU import to_hu
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--testcsv', type=str, default="")
parser.add_argument('--lowDose', type=str, required=True, help="Path to lowDose dataset folder")
parser.add_argument('--fullDose', type=str, required=True, help="Path to fullDose dataset folder")
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--depth_size', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('-w', '--weightfile', type=str, default="model/model_128.pt")
args = parser.parse_args()

testcsv = args.testcsv
lowDoseFolder = args.lowDose
fullDoseFolder = args.fullDose
input_size = args.input_size
depth_size = args.depth_size
batchsize = args.batchsize
weightfile = args.weightfile
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
in_channels = 2
out_channels = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size=input_size,
    depth_size=depth_size,
    timesteps=args.timesteps,
    loss_type='l1',
    with_condition=True,
).to(device)

diffusion.load_state_dict(torch.load(weightfile, map_location=device)['ema'])
print("Model Loaded!")
print("Weights: ", weightfile)

test_dataset = NIFTIDataset(
    testcsv, 
    fullDoseFolder,
    lowDoseFolder,
    input_size=input_size,
    depth_size=depth_size,
    augment=None  # No augmentation during inference
)

exportfolder = os.path.join("samples", weightfile)
os.makedirs(exportfolder, exist_ok=True)
print(f"Saving samples to {exportfolder}")

model.eval()
total_loss, total_psnr, total_ssim, total_input_loss = 0, 0, 0, 0
num_img = 0

# Create metrics file
metrics_file = os.path.join(exportfolder, f'metricsTestset.csv')

# Write CSV header
with open(metrics_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Sample', 'MSE_Loss (Gen)','MSE_Loss (Input)', 'PSNR', 'SSIM'])

with torch.no_grad():
    for i, data in enumerate(test_dataset):
        print(f"Processing image {i+1}/{len(test_dataset)}")
        input_tensors = data['input'].to(device).unsqueeze(0)
        target_tensors = data['target'].to(device).unsqueeze(0)
        affine = data['new_affine'].numpy()
        header = data['new_header']
        scanID = data['scan_id']
        # Get sample ID if available, otherwise use index
        sample_id = data.get('id', f"sample_{i+1}")

        generated = diffusion.sample(batch_size=input_tensors.shape[0], condition_tensors=input_tensors)

        gen_loss = F.mse_loss(generated, target_tensors)
        input_loss = F.mse_loss(input_tensors, target_tensors)
        total_loss += gen_loss.item()
        total_input_loss += input_loss.item()

        output_np = generated.cpu().numpy()
        target_np = target_tensors.cpu().numpy()
        input_np = input_tensors.cpu().numpy()

        psnr_slices, ssim_slices = [], []
        for z in range(output_np.shape[1]):
            psnr_value = skimage.metrics.peak_signal_noise_ratio(target_np[0, z], output_np[0, z])
            ssim_value = skimage.metrics.structural_similarity(target_np[0, z], output_np[0, z], multichannel=False, data_range=2.0)
            psnr_slices.append(psnr_value)
            ssim_slices.append(ssim_value)

        vol_psnr = np.mean(psnr_slices)
        vol_ssim = np.mean(ssim_slices)
        total_psnr += np.mean(psnr_slices)
        total_ssim += np.mean(ssim_slices)
        num_img += 1

        print(f"Sample {num_img} - PSNR: {np.mean(psnr_slices):.4f}, SSIM: {np.mean(ssim_slices):.4f}")

        
        sample_image = output_np.squeeze()
        input_image = input_np.squeeze()
        target_image = target_np.squeeze()

        
        target_hu = to_hu(target_image).astype(np.float32)
        generated_hu = to_hu(sample_image).astype(np.float32)
        input_hu = to_hu(input_image).astype(np.float32)

        header['scl_slope'] = 1.0
        header['scl_inter'] = 0.0

        # Save as NIfTI files
        nib.save(
            nib.Nifti1Image(generated_hu.copy(), affine=affine.copy(), header=header.copy()), 
            os.path.join(exportfolder, f'testimgGenerated-{num_img}-scanID-{scanID}.nii')
        )
        nib.save(
            nib.Nifti1Image(input_hu.copy(), affine=affine.copy(), header=header.copy()), 
            os.path.join(exportfolder, f'testimgInput-{num_img}-scanID-{scanID}.nii')
        )
        nib.save(
            nib.Nifti1Image(target_hu.copy(), affine=affine.copy(), header=header.copy()), 
            os.path.join(exportfolder, f'testimgTarget-{num_img}-scanID-{scanID}.nii')
        )
        print(f"Saved Generated, Input, and Target images for sample {num_img}")
        
        
        # --- Compute error volumes ---
        error_target_generated = np.clip(np.round((target_hu.copy() - generated_hu.copy()).astype(np.float32), 2), -3000, 3000)
        error_target_input = np.clip(np.round((target_hu.copy() - input_hu.copy()).astype(np.float32), 2), -3000, 3000)

        # Save error volumes as NIfTI
        nib.save(
            nib.Nifti1Image(error_target_generated, affine=affine.copy(), header=header.copy()),
            os.path.join(exportfolder, f'errorTargetGenerated-{num_img}-scanID-{scanID}.nii')
        )
        nib.save(
            nib.Nifti1Image(error_target_input, affine=affine.copy(), header=header.copy()),
            os.path.join(exportfolder, f'errorTargetInput-{num_img}-scanID-{scanID}.nii')
        )
        
        print(f"Saved Generated, Input, Target images and Error maps for sample {num_img}")
        
        # Save metrics for this sample
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([sample_id, f"{gen_loss.item():.6f}", f"{input_loss.item():.6f}", f"{vol_psnr:.6f}", f"{vol_ssim:.6f}"])


# Calculate and display average metrics
if num_img > 0:
    avg_test_loss = total_loss / num_img
    avg_input_loss = total_input_loss / num_img
    avg_psnr = total_psnr / num_img
    avg_ssim = total_ssim / num_img
    print(f"Average Test Loss (Gen): {avg_test_loss:.6f}, Input Loss: {avg_input_loss:.6f}")
    print(f"Average PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}")
    
    # Append overall metrics to the same CSV file
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])  # Empty row as separator
        writer.writerow(['SUMMARY', '', '', '', ''])
        writer.writerow(['Model', weightfile, '', '', ''])
        writer.writerow(['Num Samples', num_img, '', '', ''])
        writer.writerow(['Avg MSE Loss (Gen)', f"{avg_test_loss:.6f}", '', '', ''])
        writer.writerow(['Avg MSE Loss (Input)', f"{avg_input_loss:.6f}", '', '', ''])
        writer.writerow(['Avg PSNR', f"{avg_psnr:.6f}", '', '', ''])
        writer.writerow(['Avg SSIM', f"{avg_ssim:.6f}", '', '', ''])
    
    print(f"All metrics saved to {metrics_file}")