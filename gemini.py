import numpy as np
from scipy.ndimage import rotate
from typing import List
import os
import imageio
import matplotlib.pyplot as plt
import pydicom # You'll need this

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


# Specify your DICOM directory
dicom_dir = os.path.join("RadCTTACEomics_1943", "1943", "31_EQP_Ax5.00mm")

# Load and process DICOM files
print("Loading DICOM files...")
dicom_files = load_ct_images(dicom_dir)
print(f"Loaded {len(dicom_files)} DICOM files")

print("Sorting DICOM files...")
sorted_files = sort_ct(dicom_files)

print("Creating 3D volume...")
ct_volume = arrange_pixel_arrays(sorted_files)
print(f"Volume shape: {ct_volume.shape}")
    

frames = []
angles_to_rotate = np.arange(0, 360, step=10) # e.g., 0, 10, 20,... 350 degrees

# Determine a good fill value (e.g. air in CT, typically around -1000 HU)
fill_value = np.min(ct_volume) # A safe bet for CT data

for angle in angles_to_rotate:
    print(f"Processing angle: {angle}°")
    # 1. ROTATION: Rotate the 3D volume around its Z-axis (axis 0).
    #    `axes=(1, 2)` means rotate in the YX-plane.
    rotated_volume = rotate(ct_volume, angle, axes=(1, 2), reshape=False, mode='constant', cval=fill_value, order=1)
    # `rotated_volume` still has shape (Z, Y, X) but its contents are rotated.

    # 2. MIP: Perform Maximum Intensity Projection on the rotated volume.
    #    Project along the Y-axis (axis 1) of the rotated volume.
    #    This gives a view from the "side" of the spinning object.
    mip = np.max(rotated_volume, axis=1)
    # `mip` will have shape (Z, X), e.g., (60, 128) in the dummy example

    frames.append(mip)

# Get pixel spacing information for proper aspect ratios
ps = sorted_files[0].PixelSpacing
ss = sorted_files[0].SliceThickness
ax_aspect = ps[1] / ps[0]
sag_aspect = ps[1] / ss
cor_aspect = ss / ps[0]
    

# --- Step 3: Create GIF ---
gif_frames_for_imageio = []
# Determine global min/max for consistent windowing across all frames for the CT data
# This helps avoid flickering brightness/contrast in the GIF.
# You might want to use specific CT window/level values (e.g., for soft tissue, lung, bone)
global_min = np.min(ct_volume) # Or a fixed HU like -1000 (air)
global_max = np.max(ct_volume) # Or a fixed HU like +1000 (bone)
# For better visualization, often a smaller window is used, e.g., soft tissue: W=400, L=40
vmin_display = -200 # Example: Lower bound for soft tissue window
vmax_display = 200  # Example: Upper bound for soft tissue window

for i, mip_frame in enumerate(frames):
    plt.figure(figsize=(6, 6)) # Adjust aspect ratio
    plt.imshow(mip_frame, cmap='gray', origin='lower', vmin=vmin_display, vmax=vmax_display, aspect=cor_aspect)
    plt.title(f"Angle: {angles_to_rotate[i]}°")
    plt.xlabel("Rotated X-dimension")
    plt.ylabel("Z-dimension (Slices)")
    plt.axis('on') # Turn on axis to see dimensions, or 'off' for cleaner image

    fig = plt.gcf()
    fig.canvas.draw()
    
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    
    # Convert ARGB to RGB
    image_from_plot = buf[:, :, 1:4]
    
    gif_frames_for_imageio.append(image_from_plot)
    plt.close(fig)

output_gif_path = 'rotating_mip_ct_volume.gif'
imageio.mimsave(output_gif_path, gif_frames_for_imageio, fps=10) # Adjust fps as needed
print(f"GIF saved as {output_gif_path}")