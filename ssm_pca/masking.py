
import numpy as np
import nibabel as nb


def mask_img(img, mask_file):
    """ Apply mask to image"""
    mask = nb.load(mask_file).get_fdata()
    img_masked = img * mask
    return img_masked

def create_grey_matter_mask(file_list, threshold=0.38, img_preprocess=None, save_name=None):

    # Assuming all files in the directory are NIfTI files
    images = []
    for filename in file_list:
        img = nb.load(filename)
        array = img.get_fdata()
        if img_preprocess:
            array = img_preprocess(array)
        images.append(array)

    # Create individual masks
    individual_masks = []

    for img_data in images:
        max_value = np.max(img_data)
        mask = (img_data >= threshold * max_value).astype(int)
        individual_masks.append(mask)

    # Multiply all individual masks together
    multiplicative_mask = np.ones_like(individual_masks[0])

    for mask in individual_masks:
        multiplicative_mask *= mask

    # Flip the multiplicative mask left to right (mirror image)
    flipped_mask = np.flip(multiplicative_mask, axis=0)  # Assuming left-right flip along the first axis

    # Multiply the flipped mask with the original to create the final symmetrized mask
    final_mask = multiplicative_mask * flipped_mask

    if save_name:
        nb.save(nb.Nifti1Image(final_mask.astype('uint8'), img.affine), save_name)

    return final_mask


