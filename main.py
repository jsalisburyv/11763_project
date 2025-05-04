import os
import numpy as np
import pydicom
from matplotlib import pyplot as plt, animation
import scipy
from typing import List

def load_ct_images(directory_path: str) -> List[pydicom.dataset.FileDataset]:
    """Load DICOM files from directory."""
    files = []
    for fname in os.listdir(directory_path):
        files.append(pydicom.dcmread(os.path.join(directory_path, fname)))
    return files

def sort_ct(dicom_files: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
    """Sort DICOM slices by position."""
    sorting_methods = [
        lambda ds: float(ds.ImagePositionPatient[2]) if hasattr(ds, 'ImagePositionPatient') else None,
        lambda ds: float(ds.SliceLocation) if hasattr(ds, 'SliceLocation') else None,
        lambda ds: int(ds.InstanceNumber) if hasattr(ds, 'InstanceNumber') else None,
        lambda ds: int(ds.AcquisitionNumber) if hasattr(ds, 'AcquisitionNumber') else None
    ]
    
    # Try each sorting method in order until one works
    for sort_key in sorting_methods:
        try:
            sorted_datasets = sorted([ds for ds in dicom_files if sort_key(ds) is not None], 
                                    key=sort_key)
            if len(sorted_datasets) == len(dicom_files):
                return sorted_datasets
        except Exception:
            continue
    return dicom_files

def arrange_pixel_arrays(sorted_dicom_files: List[pydicom.dataset.FileDataset]) -> np.ndarray:
    """Create 3D volume from DICOM files with [z, y, x] ordering."""
    first_slice = sorted_dicom_files[0]
    pixel_shape = first_slice.pixel_array.shape
    
    # Create volume with [z, y, x] ordering
    volume = np.zeros((len(sorted_dicom_files), *pixel_shape), dtype=np.float32)
    
    for i, dicom_slice in enumerate(sorted_dicom_files):
        pixel_array = dicom_slice.pixel_array.astype(np.float32)
        # Store at z-index i
        volume[i, :, :] = pixel_array
    
    return volume

def create_rotations(volume: np.ndarray, sorted_files: List[pydicom.dataset.FileDataset], num_frames: int = 10):
    angles = np.linspace(0, 360, num_frames, endpoint=False)

    for angle in angles:
        rotated_vol = scipy.ndimage.rotate(volume, angle, axes=(1, 2), reshape=False)
        views = {
            'Axial (Z)': np.max(rotated_vol, axis=0),
            'Coronal (Y)': np.max(rotated_vol, axis=1),
            'Sagittal (X)': np.max(rotated_vol, axis=2),
        }

        for name, mip in views.items():
            plt.figure()
            plt.imshow(mip, cmap='gray')
            plt.title(name)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

def main():
    global current_dicom_files
    dicom_dir = os.path.join("RadCTTACEomics_1943", "1943", "31_EQP_Ax5.00mm")
    
    dicom_files = load_ct_images(dicom_dir)
    sorted_files = sort_ct(dicom_files)
    current_dicom_files = sorted_files
    volume = arrange_pixel_arrays(sorted_files)
    create_rotations(volume, sorted_files)

if __name__ == "__main__":
    main()
