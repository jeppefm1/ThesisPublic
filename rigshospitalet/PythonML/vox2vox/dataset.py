#AI utilized
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch
import pandas as pd
import os
import torchio as tio
import glob

class NIFTIDataset(Dataset):
    def __init__(self, csv_path, full_dose_dir, low_dose_dir, input_size, depth_size, input_channel=1, combine_output=False, augment=None):
        """
        Dataset for loading paired high and low dose NIFTI images.
        
        Args:
            csv_path: Path to CSV file containing scan IDs
            full_dose_dir: Directory containing full dose images (Z:/jeppes_project/data/reconstructions/OSSARTFull)
            low_dose_dir: Directory containing low dose images (Z:/jeppes_project/data/reconstructions/OSSART10pct)
            input_size: Height and width of the input images
            depth_size: Number of slices in the z-dimension
            input_channel: Number of input channels
            combine_output: Whether to combine input and target into a single output
            augment: Optional data augmentation transforms
        """
        self.data = pd.read_csv(csv_path)
        self.full_dose_dir = full_dose_dir
        self.low_dose_dir = low_dose_dir
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.augment = augment
        self.combine_output = combine_output

        # Find all available scan IDs by searching through subfolders
        self.pairs = self._find_paired_scans()
        
        print(f"Found {len(self.pairs)} paired scans.")

    def _find_paired_scans(self):
        """
        Match full dose and low dose images by finding scan IDs that exist in both directories.
        Search through all subfolders to locate the scan IDs.
        """
        # Get all scan IDs from the CSV
        scan_ids = self.data['Reconstruction ID'].unique()
        
        # Initialize dictionaries to store paths for each scan ID
        fd_paths = {}
        ld_paths = {}
        
        # Find full dose images
        for scan_id in scan_ids:
            # Search through all subfolders for the scan ID
            fd_matches = glob.glob(os.path.join(self.full_dose_dir, "**", f"{scan_id}.nii"), recursive=True)
            if fd_matches:
                fd_paths[scan_id] = fd_matches[0]
        
        # Find low dose images
        for scan_id in scan_ids:
            # Search through all subfolders for the scan ID
            ld_matches = glob.glob(os.path.join(self.low_dose_dir, "**", f"{scan_id}.nii"), recursive=True)
            if ld_matches:
                ld_paths[scan_id] = ld_matches[0]
        
        # Create pairs where both full dose and low dose exist
        pairs = []
        for scan_id in scan_ids:
            if scan_id in fd_paths and scan_id in ld_paths:
                pairs.append((scan_id, fd_paths[scan_id], ld_paths[scan_id]))
        
        return pairs


        nifti_img = nib.load(file_path)
        volume = nifti_img.get_fdata().astype(np.float32)
        
        # Normalize to [-1, 1] range
        volume_min = -1000
        volume_max = 2000
        volume = 2 * ((np.clip(volume, volume_min, volume_max) - volume_min) / 
                      (volume_max - volume_min)) - 1
        
        return torch.tensor(volume, dtype=torch.float32)

    def resize_img(self, img):
        """
        Resize image to the specified dimensions.
        """
        # Check if dimensions need to be transposed based on shape
        if img.dim() == 3:
            h, w, d = img.shape
        else:  # Handle 4D input (with channel dimension)
            _, h, w, d = img.shape
            
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            if img.dim() == 3:
                img = tio.ScalarImage(tensor=img[None, ...])
            else:
                img = tio.ScalarImage(tensor=img)
                
            transform = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = transform(img).data
            
            if img.dim() == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
                
        return img

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        scan_id, fd_path, ld_path = self.pairs[idx]

        # Load original NIfTI images
        fd_nifti = nib.load(fd_path)
        ld_nifti = nib.load(ld_path)

        full_dose_data = fd_nifti.get_fdata().astype(np.float32)
        low_dose_data = ld_nifti.get_fdata().astype(np.float32)

        # Get original voxel size and shape
        original_voxel_size = np.array(fd_nifti.header.get_zooms())  # (x, y, z)
        original_shape = np.array(full_dose_data.shape)              # (H, W, D)

        # Compute physical size in mm
        physical_size_mm = original_shape * original_voxel_size

        # Compute new voxel size after resizing
        new_shape = np.array([self.input_size, self.input_size, self.depth_size])
        new_voxel_size = physical_size_mm / new_shape  # (x, y, z)

        # Create new affine matrix based on new voxel size
        new_affine = np.copy(fd_nifti.affine)
        new_affine[0, 0] = new_voxel_size[0]
        new_affine[1, 1] = new_voxel_size[1]
        new_affine[2, 2] = new_voxel_size[2]

        # Clone and update header
        new_header = fd_nifti.header.copy()
        new_header.set_zooms(tuple(new_voxel_size))
        new_header['cal_min'] = -1000
        new_header['cal_max'] = 2000
        
        # Normalize image intensities to [-1, 1]
        volume_min = -1000
        volume_max = 2000
        full_dose_data = 2 * ((np.clip(full_dose_data, volume_min, volume_max) - volume_min) / (volume_max - volume_min)) - 1
        low_dose_data = 2 * ((np.clip(low_dose_data, volume_min, volume_max) - volume_min) / (volume_max - volume_min)) - 1

        # Convert to torch tensors
        full_dose = torch.tensor(full_dose_data, dtype=torch.float32)
        low_dose = torch.tensor(low_dose_data, dtype=torch.float32)
        
        # Resize images
        full_dose = self.resize_img(full_dose)
        low_dose = self.resize_img(low_dose)
        
        # Add channel dimension if needed
        if full_dose.dim() == 3:
            full_dose = full_dose.unsqueeze(0)
            low_dose = low_dose.unsqueeze(0)
        
        # Apply augmentations if specified
        if self.augment:
            subject = tio.Subject(
                low_dose=tio.ScalarImage(tensor=low_dose),
                full_dose=tio.ScalarImage(tensor=full_dose)
            )
            subject = self.augment(subject)
            
            low_dose = subject.low_dose.data
            full_dose = subject.full_dose.data
        
        if self.combine_output:
            return torch.cat([full_dose, low_dose], 0)
        
        return {
        'input': low_dose,
        'target': full_dose,
        'scan_id': scan_id,
        'new_voxel_size': torch.tensor(new_voxel_size, dtype=torch.float32),
        'new_affine': torch.tensor(new_affine, dtype=torch.float32),
        'new_header': new_header
    }
