
import numpy as np
import nibabel as nib


def preprocess(array, ref_region_mask_file='/home/leonor/Code/masks/SPM_masks/brainmask.nii'):
    """Function to normalize the images by dividing by the whole brain uptake."""

    array = np.pad(array, ((6, 6), (7, 7), (11, 11)))

    array = np.float64(array)
    mask = nib.load(ref_region_mask_file).get_fdata()

    # Mask array
    masked = array * mask

    # Select only non-zero values of array and flatt it
    masked = np.ndarray.flatten(masked)
    masked = masked[np.where(masked != 0)]

    # Get mean of ref region
    mean_ref = np.mean(masked)

    array = array / mean_ref

    return array