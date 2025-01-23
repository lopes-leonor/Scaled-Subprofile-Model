
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import nibabel as nb

from ssm_pca.image_preprocess import preprocess
from ssm_pca.masking import mask_img
from ssm_pca.get_rois import get_rois


def tpr(filepath, mask_file, gis_pc, group_mean_profile, to_mni_space=True):
    """
    Returns:
        object:
    """
    img = nb.load(filepath)
    array = img.get_fdata()

    array = preprocess(array, to_mni_space=to_mni_space)
    
    mask = nb.load(mask_file).get_fdata()

    # Get mask in row format
    mask_row = mask.flatten()

    # Mask with given file
    #array = mask_img(array, mask_file=mask_file)
    array = array * mask

    vector = array.flatten()

    # Change the mask so that 0 value of subject is removed   
    mask_row = mask_row*(vector > 0)
    mask_sum = np.sum(mask_row)

    # Replace all values <= 0 with 1 (to be able to do the log)
    vector[vector <= 0] = 1

    # Log transform vector
    log_vector = np.log(vector)

    # Subtract mean of subject
    subject_mean = np.mean(vector)
    centered_vector = log_vector - subject_mean

    # Mask again
    centered_vector = mask_row * centered_vector

    # Compute subject residual profiles (SRP) - subtract group mean profile
    srp = centered_vector - group_mean_profile

    score = np.dot(srp, gis_pc)

    return srp, score

def tpr_roi(filepath, mask_file, roi_mask_file, gis_pc, group_mean_profile, to_mni_space=True):
    """

    Returns:
        object:
    """
    img = nb.load(filepath)
    array = img.get_fdata()

    if to_mni_space:
        array = np.pad(array, ((6, 6), (7, 7), (11, 11)))

    array = mask_img(array, mask_file=mask_file)

    roi_mask = nb.load(roi_mask_file).get_fdata()

    array, labels = get_rois(array, rois_mask=roi_mask)

    array = array / np.mean(array)
    # array = (array - np.mean(array)) / np.std(array)

    # Log transform group matrix
    array = np.log(array)

    subject_mean = np.mean(array)

    vector = array

    srp = vector - subject_mean - group_mean_profile

    score = np.dot(srp, gis_pc)

    return srp, score


def prospective_scores(filelist, mask_file, gis_pc_file, group_mean_profile_file, mean_normals_file, to_mni_space=True, save_dir=None):

    gis_pc = np.load(gis_pc_file)
    gmp = np.load(group_mean_profile_file)
    mean_normals = np.load(mean_normals_file)
    scores = []

    for filepath in filelist:
        _, score = tpr(filepath, mask_file, gis_pc, gmp, to_mni_space)
        scores.append(score)

    # Z-score transformation
    scores_z = (np.array(scores) - np.mean(scores)) / np.std(scores)

    # Offset by controls mean
    if mean_normals:
        scores_z = scores_z - mean_normals

    if save_dir:
        # Save the scores into a dataframe
        data = {'filepaths': filelist, 'scores': scores_z}
        df = pd.DataFrame(data)
        df.to_csv(save_dir + '/scores_prospective.csv')


if __name__ == '__main__':
    #Parse config file path

    parser = ArgumentParser()
    parser.add_argument("--csv_file", type=str, help='CSV file with filepaths and corresponding labels of normal controls (0) or diseased subjects (1). Columns should be filepaths and labels.')
    parser.add_argument("--mask_file", type=str, help='Path to a binary brain mask file to apply to the images.')
    parser.add_argument("--gis_pc_file", type=str, help='Path to the GIS pattern (.npy) you want to get the scores from.')
    parser.add_argument("--gmp", type=str, help='Path to the group mean profile file (.npy).')
    parser.add_argument("--mean_normals_file", type=str, help='Path to the mean of control group (.npy).')
    parser.add_argument("--save_dir", type=str, help='Directory path where to save the results files.')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_file)

    filelist = list(df['filepaths'])


    prospective_scores(filelist, mask_file=args.mask_file, gis_pc_file=args.gis_pc_file, group_mean_profile_file=args.gmp, 
                      mean_normals_file=args.mean_normals_file, to_mni_space=True, save_dir=args.save_dir)
 