import os
import numpy as np
import pydicom
from matplotlib import pyplot as plt, animation
import scipy.ndimage
from typing import List, Dict, Any
import matplotlib
# Use a non-interactive backend for better GIF saving
matplotlib.use('Agg')

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

def create_mip_gifs(volume: np.ndarray, sorted_files: List[pydicom.dataset.FileDataset], num_frames: int = 36, output_dir: str = "output"):
    """Create rotating Maximum Intensity Projection GIFs for each view."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get pixel spacing information for proper aspect ratios
    ps = sorted_files[0].PixelSpacing
    ss = sorted_files[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]
    
    # Create angles for rotation
    angles = np.linspace(0, 360, num_frames, endpoint=False)
    
    # Define the views
    views = ['axial', 'coronal', 'sagittal']
    
    # Create all rotated volumes first to avoid redundant calculations
    print("Generating rotated volumes...")
    rotated_volumes = []
    for angle in angles:
        rotated_vol = scipy.ndimage.rotate(volume, angle, axes=(1, 2), reshape=False)
        rotated_volumes.append(rotated_vol)
    
    # Process each view separately
    for view_idx, view_name in enumerate(views):
        print(f"Processing {view_name} view...")
        
        # Generate all MIPs for this view
        mips = []
        for rotated_vol in rotated_volumes:
            if view_name == 'axial':
                mip = np.max(rotated_vol, axis=0)
            elif view_name == 'coronal':
                mip = np.max(rotated_vol, axis=1)
                mip = np.flipud(mip)  # Flip for proper orientation
            else:  # sagittal
                mip = np.max(rotated_vol, axis=2)
                mip = np.flipud(mip)  # Flip for proper orientation
            mips.append(mip)
        
        # Create a new figure for the animation
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.tight_layout()
        ax.set_title(f"Rotating MIP - {view_name.capitalize()} View")
        ax.axis('off')
        
        # Set appropriate aspect ratio
        if view_name == 'axial':
            ax.set_aspect(ax_aspect)
        elif view_name == 'coronal':
            ax.set_aspect(cor_aspect)
        else:  # sagittal
            ax.set_aspect(1/sag_aspect)
        
        # Initialize with the first image
        img = ax.imshow(mips[0], cmap='gray', animated=True)
        
        # Animation update function
        def update(frame):
            img.set_array(mips[frame])
            return [img]
        
        # Create animation using FuncAnimation instead of ArtistAnimation
        anim = animation.FuncAnimation(
            fig, update, frames=len(mips), interval=50, blit=True
        )
        
        # Save as GIF with a different writer
        print(f"Saving {view_name} MIP rotation GIF...")
        
        # Try using different writers based on what's available
        try:
            # Try using imagemagick first
            anim.save(
                f"{output_dir}/{view_name}_mip_rotation.gif", 
                writer='imagemagick', 
                fps=10,
                dpi=100
            )
        except:
            try:
                # If imagemagick fails, try pillow
                from PIL import Image
                
                # Save individual frames
                frames_dir = os.path.join(output_dir, "temp_frames")
                if not os.path.exists(frames_dir):
                    os.makedirs(frames_dir)
                
                print("Saving individual frames...")
                frame_files = []
                for i, mip in enumerate(mips):
                    # Create a new figure for each frame to avoid issues
                    frame_fig, frame_ax = plt.subplots(figsize=(8, 8))
                    frame_ax.axis('off')
                    
                    # Set appropriate aspect ratio
                    if view_name == 'axial':
                        frame_ax.set_aspect(ax_aspect)
                    elif view_name == 'coronal':
                        frame_ax.set_aspect(cor_aspect)
                    else:  # sagittal
                        frame_ax.set_aspect(1/sag_aspect)
                    
                    frame_ax.imshow(mip, cmap='gray')
                    
                    # Save the frame
                    frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
                    frame_files.append(frame_path)
                    frame_fig.savefig(frame_path, bbox_inches='tight', pad_inches=0)
                    plt.close(frame_fig)
                
                # Create GIF from saved frames using PIL
                print("Creating GIF from frames...")
                frames = [Image.open(f) for f in frame_files]
                frames[0].save(
                    f"{output_dir}/{view_name}_mip_rotation.gif",
                    save_all=True,
                    append_images=frames[1:],
                    duration=100,
                    loop=0
                )
                
                # Clean up temporary files
                for f in frame_files:
                    os.remove(f)
                if os.path.exists(frames_dir):
                    os.rmdir(frames_dir)
            except Exception as e:
                print(f"Error creating GIF: {e}")
                # If all else fails, just save the first frame as a static image
                plt.imsave(f"{output_dir}/{view_name}_mip_static.png", mips[0], cmap='gray')
        
        # Close the main figure
        plt.close(fig)
        print(f"Created {view_name} MIP rotation GIF")

def main():
    # Specify your DICOM directory
    dicom_dir = os.path.join("RadCTTACEomics_1943", "1943", "31_EQP_Ax5.00mm")
    
    # Load and process DICOM files
    print("Loading DICOM files...")
    dicom_files = load_ct_images(dicom_dir)
    print(f"Loaded {len(dicom_files)} DICOM files")
    
    print("Sorting DICOM files...")
    sorted_files = sort_ct(dicom_files)
    
    print("Creating 3D volume...")
    volume = arrange_pixel_arrays(sorted_files)
    print(f"Volume shape: {volume.shape}")
    
    # Create MIP GIFs
    # create_mip_gifs(volume, sorted_files)
    # print("All MIP rotation GIFs created successfully!")

if __name__ == "__main__":
    main()
