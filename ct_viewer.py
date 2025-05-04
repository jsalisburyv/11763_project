import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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
    """Create 3D volume from DICOM files."""
    first_slice = sorted_dicom_files[0]
    pixel_shape = first_slice.pixel_array.shape
    volume = np.zeros((*pixel_shape, len(sorted_dicom_files)), dtype=np.float32)
    
    for i, dicom_slice in enumerate(sorted_dicom_files):
        pixel_array = dicom_slice.pixel_array.astype(np.float32)
        volume[:, :, i] = pixel_array
    return volume

def view_all_planes(volume: np.ndarray) -> None:
    """Display orthogonal planes of the volume with interactive sliders."""
    y_dim, x_dim, z_dim = volume.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    axes[1, 1].axis('off')
    
    first_slice = None
    try:
        first_slice = current_dicom_files[0]
    except:
        pass
            
    ax_aspect = 1.0
    sag_aspect = 1.0
    cor_aspect = 1.0
    
    if first_slice is not None and hasattr(first_slice, 'PixelSpacing') and hasattr(first_slice, 'SliceThickness'):
        ps = first_slice.PixelSpacing
        ss = first_slice.SliceThickness
        ax_aspect = ps[1] / ps[0]
        sag_aspect = ps[1] / ss
        cor_aspect = ss / ps[0]
    
    axial_idx = z_dim // 2
    coronal_idx = y_dim // 2
    sagittal_idx = x_dim // 2
    
    axial_slice = volume[:, :, axial_idx]
    
    flipped_sagittal_idx = x_dim - 1 - sagittal_idx
    sagittal_slice = volume[:, flipped_sagittal_idx, :]
    sagittal_slice = sagittal_slice.T
    sagittal_slice = np.flipud(sagittal_slice)
    
    flipped_coronal_idx = y_dim - 1 - coronal_idx
    coronal_slice = volume[flipped_coronal_idx, :, :].T
    coronal_slice = np.flipud(coronal_slice)
    
    axial_img = axes[0, 0].imshow(axial_slice, cmap='gray')
    axes[0, 0].set_title(f"Axial (z={axial_idx+1}/{z_dim})")
    axes[0, 0].set_aspect(ax_aspect)
    
    sagittal_img = axes[0, 1].imshow(sagittal_slice, cmap='gray')
    axes[0, 1].set_title(f"Sagittal (x={flipped_sagittal_idx+1}/{x_dim})")
    axes[0, 1].set_aspect(1/sag_aspect)
    
    coronal_img = axes[1, 0].imshow(coronal_slice, cmap='gray')
    axes[1, 0].set_title(f"Coronal (y={flipped_coronal_idx+1}/{y_dim})")
    axes[1, 0].set_aspect(cor_aspect)
    
    window_width = 400
    window_center = 40
    vmin, vmax = window_center - window_width/2, window_center + window_width/2
    
    axial_img.set_clim(vmin, vmax)
    sagittal_img.set_clim(vmin, vmax)
    coronal_img.set_clim(vmin, vmax)
    
    ax_slider_axial = plt.axes((0.25, 0.05, 0.65, 0.02), facecolor='lightgoldenrodyellow')
    ax_slider_coronal = plt.axes((0.25, 0.10, 0.65, 0.02), facecolor='lightgoldenrodyellow')
    ax_slider_sagittal = plt.axes((0.25, 0.15, 0.65, 0.02), facecolor='lightgoldenrodyellow')
    
    slider_axial = Slider(ax=ax_slider_axial, label='Axial', valmin=0, valmax=z_dim-1, valinit=axial_idx, valstep=1)
    slider_coronal = Slider(ax=ax_slider_coronal, label='Coronal', valmin=0, valmax=y_dim-1, valinit=coronal_idx, valstep=1)
    slider_sagittal = Slider(ax=ax_slider_sagittal, label='Sagittal', valmin=0, valmax=x_dim-1, valinit=sagittal_idx, valstep=1)
    
    def update_axial(val):
        idx = int(slider_axial.val)
        axial_img.set_data(volume[:, :, idx])
        axes[0, 0].set_title(f"Axial (z={idx+1}/{z_dim})")
        fig.canvas.draw_idle()
    
    def update_sagittal(val):
        idx = x_dim - 1 - int(slider_sagittal.val)
        sagittal_slice = volume[:, idx, :]
        sagittal_slice = sagittal_slice.T
        sagittal_slice = np.flipud(sagittal_slice)
        sagittal_img.set_data(sagittal_slice)
        axes[0, 1].set_title(f"Sagittal (x={idx+1}/{x_dim})")
        fig.canvas.draw_idle()
    
    def update_coronal(val):
        idx = y_dim - 1 - int(slider_coronal.val)
        coronal_slice = volume[idx, :, :].T
        coronal_slice = np.flipud(coronal_slice)
        coronal_img.set_data(coronal_slice)
        axes[1, 0].set_title(f"Coronal (y={idx+1}/{y_dim})")
        fig.canvas.draw_idle()
    
    slider_axial.on_changed(update_axial)
    slider_sagittal.on_changed(update_sagittal)
    slider_coronal.on_changed(update_coronal)
    
    plt.show()

current_dicom_files = []

def main():
    global current_dicom_files
    dicom_dir = os.path.join("RadCTTACEomics_1943", "1943", "31_EQP_Ax5.00mm")
    
    dicom_files = load_ct_images(dicom_dir)
    sorted_files = sort_ct(dicom_files)
    current_dicom_files = sorted_files
    volume = arrange_pixel_arrays(sorted_files)
    view_all_planes(volume)

if __name__ == "__main__":
    main()
