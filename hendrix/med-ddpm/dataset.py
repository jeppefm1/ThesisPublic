#Inspiration from https://github.com/mobaidoctor/med-ddpm
#https://torchio.readthedocs.io/
#AI utilized

from torch.utils.data import Dataset
import nibabel as nib
import pydicom
import torchio as tio
import numpy as np
import torch
import pandas as pd
import os
from utils.toHU import to_hu

class DicomDataset(Dataset):
    def __init__(self, csv_path, root_dir, input_size, depth_size, save_nifti=False, nifti_output_dir=None, input_channel: int = 1, combine_output=False, augment=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.save_nifti = save_nifti
        self.nifti_output_dir = nifti_output_dir
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.augment = augment
        self.combine_output = combine_output

        # Ensure output directory exists
        if self.save_nifti and self.nifti_output_dir:
            os.makedirs(self.nifti_output_dir, exist_ok=True)

        # Match Low Dose and Full Dose images by Subject ID
        self.pairs = self._match_series()

    def _match_series(self):
        """ Match Low Dose and Full Dose images based on Subject ID. """
        ld_dict, fd_dict = {}, {}

        for _, row in self.data.iterrows():
            subject_id, series_uid, series_desc = row['Subject ID'], row['Series UID'], row['Series Description']
            if series_desc == "Low Dose Images":
                ld_dict[subject_id] = series_uid
            elif series_desc == "Full Dose Images":
                fd_dict[subject_id] = series_uid

        # Keep only pairs where both exist
        return [(ld_dict[sid], fd_dict[sid]) for sid in ld_dict if sid in fd_dict]

    def _load_dicom_series(self, series_uid):
        """ Load and stack DICOM slices into a 3D tensor. """
        series_path = os.path.join(self.root_dir, series_uid)
        dicom_files = sorted([os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.dcm')])

        images = []
        for file in dicom_files:
            dcm = pydicom.dcmread(file)
            img = dcm.pixel_array.astype(np.float32)  # Convert to float
            images.append(img)

        # Stack slices into (num_slices, H, W) -> (D, H, W)
        volume = np.stack(images, axis=0)
        volume_min = 0.0
        volume_max = 4095.0
        volume = 2 * ((volume - volume_min) / (volume_max - volume_min)) - 1
        
        return torch.tensor(volume, dtype=torch.float32)

#Warning: In most medical image applications, this transform should not be used as it will deform the physical object by scaling anistropically along the different dimensions. The solution to change an image size is typically applying Resample and CropOrPad.

    def resize_img(self, img):
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]
        return img

    def resize_img_4d(self, input_img):
        h, w, d, c = input_img.shape
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, 2))
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                buff = input_img.copy()[..., ch]
                img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
                cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
                img = np.asarray(cop(img))[0]
                result_img[..., ch] += img
            return result_img
        else:
            return input_img

    def _save_nifti(self, img, subject_id, series_desc):
        """ Save rescaled 3D image as NIfTI. """
        hu_img = to_hu(img)
        nifti_img = nib.Nifti1Image(hu_img, affine=np.eye(4))
        nifti_filename = f"{subject_id}_{series_desc}.nii.gz"
        nifti_path = os.path.join(self.nifti_output_dir, nifti_filename)
        nib.save(nifti_img, nifti_path)
        print(f"Saved NIfTI file: {nifti_path}")


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ld_uid, fd_uid = self.pairs[idx]

        low_dose = self._load_dicom_series(ld_uid)
        full_dose = self._load_dicom_series(fd_uid)

        # Rescale to (h, w, d)
        low_dose = self.resize_img(low_dose) if self.input_channel == 1 else self.resize_img_4d(low_dose)
        full_dose = self.resize_img(full_dose) if self.input_channel == 1 else self.resize_img_4d(full_dose)

        # TorchIO and network expects a 4D tensor (C, H, W, D) 
        low_dose = torch.tensor(low_dose).unsqueeze(0)
        full_dose = torch.tensor(full_dose).unsqueeze(0)

        #Permute to C, D, H, W
        low_dose = low_dose.permute(0, 3, 1, 2)
        full_dose = full_dose.permute(0, 3, 1, 2)

        if self.augment:
            subject = tio.Subject(
                low_dose=tio.ScalarImage(tensor=low_dose),
                full_dose=tio.ScalarImage(tensor=full_dose)
            )
            subject = self.augment(subject)

            low_dose = subject.low_dose.data
            full_dose = subject.full_dose.data

        # Save as NIfTI if enabled
        if self.save_nifti:
            low_dose_save = low_dose.permute(0, 2, 3, 1)
            full_dose_save = full_dose.permute(0, 2, 3, 1)

            self._save_nifti(low_dose_save.squeeze(0).cpu().numpy(), ld_uid, "LowDose")
            self._save_nifti(full_dose_save.squeeze(0).cpu().numpy(), fd_uid, "FullDose")


        if self.combine_output:
            return torch.cat([full_dose, low_dose], 0)

        return {'input':low_dose, 'target':full_dose}

# csv_path = "../../../data/ReconstructedSiemens/metadata.csv"
# root_dir = "../../../data/ReconstructedSiemens/images"
# nifti_output_dir = "../../../scratch/nifti"



# torchio_transform = tio.Compose([
#     tio.RandomFlip(axes=['LR', 'AP', 'IS'], flip_probability=0.5),
# ])


# dataset = DicomDataset(csv_path, root_dir, input_size = 128, depth_size = 128,save_nifti=True, nifti_output_dir=nifti_output_dir, augment=torchio_transform)

# # Example: Get a sample
# dataset[0]
# dataset[1]
# dataset[2]
