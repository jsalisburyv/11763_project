import os
import numpy as np
import pydicom
import highdicom as hd
from matplotlib import pyplot as plt, animation
import scipy


def load_ct_image(path):
    dicom_files = sorted([f for f in os.listdir(path) if f.endswith('.dcm')])
    dicom_datasets = []
    
    for file in dicom_files:
        file_path = os.path.join(path, file)
        dicom_datasets.append(pydicom.dcmread(file_path))
    
    return dicom_datasets

def sort_ct_slices(dicom_datasets):
    # Define sorting functions in priority order
    sorting_methods = [
        lambda ds: float(ds.ImagePositionPatient[2]) if hasattr(ds, 'ImagePositionPatient') else None,
        lambda ds: float(ds.SliceLocation) if hasattr(ds, 'SliceLocation') else None,
        lambda ds: int(ds.InstanceNumber) if hasattr(ds, 'InstanceNumber') else None,
        lambda ds: int(ds.AcquisitionNumber) if hasattr(ds, 'AcquisitionNumber') else None
    ]
    
    # Try each sorting method in order until one works
    for sort_key in sorting_methods:
        try:
            # Filter out any None values from datasets where the attribute doesn't exist
            sorted_datasets = sorted([ds for ds in dicom_datasets if sort_key(ds) is not None], 
                                    key=sort_key)
            # If we got the same number of datasets back, the sort was successful
            if len(sorted_datasets) == len(dicom_datasets):
                return sorted_datasets
        except Exception:
            continue
    # If all sorting methods fail, return the original order
    print("Warning: Could not determine reliable slice order from DICOM headers")
    return dicom_datasets


def arrange_pixel_arrays(dicom_datasets):
    # Get dimensions from the first slice
    num_slices = len(dicom_datasets)
    rows = dicom_datasets[0].Rows
    cols = dicom_datasets[0].Columns
    
    # Initialize 3D array to store the volume
    volume = np.zeros((num_slices, rows, cols), dtype=np.int16)
    
    # Fill the volume with pixel data from each slice
    for i, ds in enumerate(dicom_datasets):
        volume[i, :, :] = ds.pixel_array
    
    return volume

def load_segmentation(file_path):
    seg = hd.seg.segread(file_path)
    return seg

def extract_segment_arrays(segmentation):

    # Get the segment numbers 
    segment_numbers = segmentation.segment_numbers
    
    # Dictionary to store segment masks by segment number
    segment_masks = {}
    
    # Get source image UIDs
    source_image_uids = segmentation.get_source_image_uids()
    if not source_image_uids:
        print("No source images found in segmentation")
        return segment_masks
    
    # Extract source SOP instance UIDs
    source_sop_instance_uids = [uid[2] for uid in source_image_uids]
    
    # Try using get_segment_mask - this is the method shown in the quickstart guide
    for segment_number in segment_numbers:
        try:
            # Follow the quickstart guide approach
            mask = None
            
            # First try get_pixel_array if available
            if hasattr(segmentation, 'pixel_array'):
                # If the segmentation has a pixel_array attribute, use it directly
                mask = segmentation.pixel_array
                if mask.ndim > 3 and mask.shape[-1] > 1:
                    # If it's a multi-segment array, extract the specific segment
                    segment_idx = segment_numbers.index(segment_number)
                    mask = mask[..., segment_idx]
            else:
                # Try using the approach from the documentation
                try:
                    # Try get_pixels_by_source_instance method
                    pixels = segmentation.get_pixels_by_source_instance(
                        source_sop_instance_uids=source_sop_instance_uids,
                        segment_numbers=[segment_number]
                    )
                    
                    # If we have pixels, use them
                    if pixels is not None:
                        if pixels.ndim > 3:
                            # Squeeze out the last dimension if it exists
                            mask = np.squeeze(pixels, axis=-1)
                        else:
                            mask = pixels
                except Exception as e:
                    print(f"Error getting pixels by source instance: {e}")
                    
                    # If that fails, try the get_array method if it exists
                    if hasattr(segmentation, 'get_array'):
                        try:
                            mask = segmentation.get_array(segment_number)
                        except Exception as e2:
                            print(f"Error getting array: {e2}")
                
            # If we got a mask, store it
            if mask is not None:
                segment_masks[segment_number] = mask
                
        except Exception as e:
            print(f"Error extracting segment {segment_number}: {e}")
    
    return segment_masks

def align_segmentation_with_ct(ct_datasets, segmentation):
    """
    Align segmentation masks with CT datasets using spatial coordinates.
    
    Args:
        ct_datasets: List of sorted CT DICOM datasets
        segmentation: A highdicom segmentation object
        
    Returns:
        Dictionary mapping segment numbers to aligned 3D mask arrays
    """
    # Get dimensions from the CT volume
    num_slices = len(ct_datasets)
    rows = ct_datasets[0].Rows
    cols = ct_datasets[0].Columns
    
    # Create empty aligned segments
    aligned_segments = {}
    for segment_number in segmentation.segment_numbers:
        aligned_segments[segment_number] = np.zeros((num_slices, rows, cols), dtype=bool)
    
    try:
        # Simple approach: if the frames of the segmentation directly correspond to frames of the source images,
        # we can try to extract them based on source image UIDs
        
        # Get source image references
        source_image_references = segmentation.get_source_image_uids()
        
        # Try to extract segment arrays
        segment_masks = extract_segment_arrays(segmentation)
        
        # If segment extraction worked, try to align them
        if segment_masks:
            for segment_number, mask in segment_masks.items():
                # If the mask already has the right shape, use it directly
                if mask.shape == (num_slices, rows, cols):
                    aligned_segments[segment_number] = mask.astype(bool)
                elif len(mask.shape) == 3:
                    # Otherwise, try to resize the mask to match the CT volume
                    temp_mask = np.zeros((num_slices, rows, cols), dtype=bool)
                    
                    # Copy the mask data, handling size differences
                    min_slices = min(num_slices, mask.shape[0])
                    min_rows = min(rows, mask.shape[1])
                    min_cols = min(cols, mask.shape[2])
                    
                    # Copy the data we have
                    temp_mask[:min_slices, :min_rows, :min_cols] = mask[:min_slices, :min_rows, :min_cols] > 0
                    
                    aligned_segments[segment_number] = temp_mask
                else:
                    print(f"Cannot align segment {segment_number}: unexpected shape {mask.shape}")
        
    except Exception as e:
        print(f"Error aligning segmentation: {e}")
        
    return aligned_segments

def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)

def generate_mip(ct_volume, plane='coronal'):
    # Select axis based on desired anatomical plane
    if plane == 'axial':
        # Project along superior-inferior axis (axis 0)
        mip = np.max(ct_volume, axis=0)
    elif plane == 'coronal':
        # Project along anterior-posterior axis (axis 1)
        mip = np.max(ct_volume, axis=1)
    elif plane == 'sagittal':
        # Project along left-right axis (axis 2)
        mip = np.max(ct_volume, axis=2)
    else:
        raise ValueError("Plane must be one of: 'axial', 'coronal', 'sagittal'")
    
    return mip


def generate_rotation_gif(ct_volume, segmentation_mask=None, plane='coronal', output_path='rotation.gif'):
    #   Create projections
    ct_volume = np.flip(ct_volume, axis=0)
    img_min = np.amin(ct_volume)
    img_max = np.amax(ct_volume)
    cm = plt.colormaps['bone']
    fig, ax = plt.subplots()
    n = 6
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360*(n-1)/n, num=n)):
        rotated_img = rotate_on_axial_plane(ct_volume, alpha)
        projection = generate_mip(rotated_img, plane)
        plt.imshow(projection, cmap=cm, vmin=img_min, vmax=img_max)#, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        #plt.savefig(f'results/MIP/Projection_{idx}.png')      # Save animation
        projections.append(projection)  # Save for later animation

    # Save and visualize animation
    animation_data = [
        [plt.imshow(img, animated=True, cmap=cm, vmin=img_min, vmax=img_max)]#, aspect=pixel_len_mm[0] / pixel_len_mm[1])]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                              interval=250, blit=True)
    anim.save(output_path)  # Save animation
    plt.show()                              # Show animation

def main():
    # Load CT image
    ct_path = os.path.join('RadCTTACEomics_1943', '1943', '31_EQP_Ax5.00mm')
    dicom_datasets = load_ct_image(ct_path)
    print(f"CT scan loaded: {len(dicom_datasets)} slices")
    sorted_datasets = sort_ct_slices(dicom_datasets)
    print("CT slices sorted by position")
    ct_volume = arrange_pixel_arrays(sorted_datasets)
    print(f"CT volume arranged: shape {ct_volume.shape}")
    
    # Load segmentation
    segmentation_path = os.path.join('RadCTTACEomics_1943', '1943', '31_EQP_Ax5.00mm_ManualROI_Tumor.dcm')
    segmentation = load_segmentation(segmentation_path)
    print(f"Segmentation loaded: shape {segmentation.segment_numbers}")

    # Align segmentation with CT
    aligned_segments = align_segmentation_with_ct(sorted_datasets, segmentation)
    print("Segmentation aligned with CT")
    
    # If there are segments, use the first one for the rotation GIF
    if aligned_segments:
        first_segment = list(aligned_segments.values())[0]
        generate_rotation_gif(ct_volume, first_segment, 'coronal', 'ct_with_segmentation_rotation.gif')
    else:
        generate_rotation_gif(ct_volume, 'coronal', 'ct_rotation.gif')
    
    print("Processing complete")

if __name__ == "__main__":
    main()
