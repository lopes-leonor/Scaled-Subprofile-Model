
import numpy as np
from ssm_pca.masking import mask_img

def get_rois(img, rois_mask, rois_list=None):
    """ Get mean values of each ROI in rois_mask or of rois_list."""
    
    # Choose either rois from a list (if you want only significant ones for e.g.) or all
    if rois_list:
        unique_labels = rois_list
    else:
        unique_labels = np.unique(rois_mask)[1:]  # Exclude background


    means_all_regions = []
    labels = []

    for label in unique_labels:
        # Create mask for the current region
        mask = rois_mask.copy()
        mask[mask != label] = 0
        mask[mask == label] = 1

        # Apply the mask to the image
        masked_img = mask_img(img, mask=mask)

        # Select only non-zero values of array and flatt it
        masked_img = np.ndarray.flatten(masked_img)
        masked_img = masked_img[np.where(masked_img != 0)]

        # Get mean of region
        mean = np.mean(masked_img)
        #std = np.std(masked_img)

        if mean == None:
            print(np.max(masked_img))

        # Append the mean and correspondent label to lists
        means_all_regions.append(mean)
        #std_all_regions.append(std)
        labels.append(label)

    return means_all_regions, labels