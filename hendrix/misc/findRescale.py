import pandas as pd
import os
import pydicom
import numpy as np

class HUValueFinder:
    def __init__(self, csv_path, root_dir):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.series_uids = self.data['Series UID'].unique()
        
    def _load_dicom_series(self, series_uid):
        """ Load and stack DICOM slices into a 3D tensor. """
        series_path = os.path.join(self.root_dir, series_uid)
        dicom_files = sorted([os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.dcm')])

        images = []
        for file in dicom_files:
            dcm = pydicom.dcmread(file)
            img = dcm.pixel_array.astype(np.float32)  # Convert to float
            images.append(img)
        
        volume = np.stack(images, axis=0)  # Stack slices into (D, H, W)
        return volume
    
    def find_min_max(self):
        global_min = float('inf')
        global_max = float('-inf')
        
        for series_uid in self.series_uids:
            volume = self._load_dicom_series(series_uid)
            series_min = volume.min()
            series_max = volume.max()
            
            global_min = min(global_min, series_min)
            global_max = max(global_max, series_max)
            
            print(f"Series: {series_uid} -> Min: {series_min}, Max: {series_max}")
        
        print("\nOverall Min and Max across all series:")
        print(f"Min: {global_min}, Max: {global_max}")
        
        return global_min, global_max

csv_path = "../../../data/ReconstructedSiemens/metadata.csv"
root_dir = "../../../data/ReconstructedSiemens/images"

finder = HUValueFinder(csv_path, root_dir)
min_HU, max_HU = finder.find_min_max()
