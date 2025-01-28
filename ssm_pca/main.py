
from argparse import ArgumentParser
import os
import numpy as np
import nibabel as nb
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt

from ssm_pca.image_preprocess import preprocess
from ssm_pca.masking import create_grey_matter_mask
from ssm_pca.pca import apply_pca, plot_eigenvalues_vaf
from ssm_pca.utils import pc_scores_to_df, display_gis
from ssm_pca.selecting_best_pc import best_pc_combination, combine_pc_vectors, compare_scores_normal_disease


def ssm_pca(filelist=None, labels=None, preprocess_img=True, mask_file=None, save_dir=None):

    """ Main function to run Scaled Subprofile Model with Principal Component Analysis on the images on filelist.

    Args:
        filelist: List with paths to the images to use. Defaults to None.
        labels: List to the labels corresponding to the images. Defaults to None.
        preprocess_img: Whether to preprocess the images. Defaults to True.
        mask_file: Path to a mask file to mask the images or threshold to create mask. If None, no mask is applied Defaults to None.
        save_dir: Directory to save the results. If None, results are not saved. Defaults to None.

    Returns:
        z_transf_gis - z_transformed_patterns: 2D numpy array of Z-transformed voxel pattern vectors (n_voxels, n_components)  
        gis - voxel pattern eigenvectors
        vaf - variace accounted for by each component
        gmp - group mean profile
    """

    
    n_subj = len(filelist)
    print(f'Number of subjects: {n_subj}')
    n_d = np.count_nonzero(labels)
    print(f'Number of diseased subjects: {n_d}')
    n_nc = n_subj - n_d
    print(f'Number of normal controls: {n_nc}')

    # Create save directories
    vector_save_dir = save_dir + '/vectors'
    os.makedirs(vector_save_dir, exist_ok=True)

    nifti_save_dir = save_dir + '/nifti'
    os.makedirs(nifti_save_dir, exist_ok=True)

    plots_save_dir = save_dir + '/plots'
    os.makedirs(plots_save_dir, exist_ok=True)


    # Load mask
    if mask_file:
        if type(mask_file) == float:
            mask = create_grey_matter_mask(filelist, threshold=float(mask_file), save_name=save_dir + '/threshold_mask.nii')
        else:
            mask = nb.load(mask_file).get_fdata()

    row_vectors_list = []

    # Load images and create row vectors
    for filepath in filelist:
        img = nb.load(filepath)
        array = img.get_fdata()

        # Preprocess image (change dimensions, spatial normalize, intensity normalize, etc)
        if preprocess_img:
            array = preprocess(array)

        # Mask with loaded mask
        if mask_file:
            array = array * mask

        # Convert the array to a row vector
        row_vector = array.flatten()

        row_vectors_list.append(row_vector)

    print('Creating group matrix...')
    # Create group matrix
    subj_matrix = np.vstack(row_vectors_list)

    # Get mask in row format for later use
    mask_row = mask.flatten()
    mask_sum = np.sum(mask_row) # Number of voxels in the mask

    # Replace all values <= 0 with 1 (to be able to do the log)
    subj_matrix[subj_matrix <= 0] = 1

    print('Log transforming data...')
    # Log transform the data
    log_data_matrix = np.log(subj_matrix)

    # Center the data matrix (subtract the row means). Get GMP - column means.
    subject_means = np.sum(log_data_matrix, axis=1)/mask_sum

    centered_data_matrix = log_data_matrix - subject_means[:, np.newaxis]

    # Mask again
    centered_data_matrix = centered_data_matrix * mask_row
  
    print('Computing SRP...')
    # Compute subject residual profiles (SRP) - subtract column means
    voxels_means = np.mean(centered_data_matrix, axis=0, keepdims=True)
    subject_residual_profiles = centered_data_matrix - voxels_means 

    if save_dir:
        np.save(vector_save_dir + '/group_mean_profile.npy', voxels_means)
  
    # Mask again 
    subject_residual_profiles = subject_residual_profiles * mask_row

    # Compute subject-by-subject covariance matrix
    covariance_matrix = subject_residual_profiles @ subject_residual_profiles.T

    print('Appliying PCA...')
    # Apply PCA to the covariance matrix
    score_vectors, eigVal, vaf = apply_pca(covariance_matrix)

    if save_dir:
        np.save(vector_save_dir + f'/score_vectors.npy', score_vectors)

    if save_dir:
        # Plot the variance accounted for (vaf) by each component
        plot_eigenvalues_vaf(eigVal, vaf)    

    # GIS vector and its negative form are mathematically equivalent solutions of the eigenvector equation 
    # with a corresponding sign flip of associated subject scores. 
    # Only the GIS form corresponding to positive mean patient scores 
    # relative to normal subject scores is considered to be relevant.

    print('Changing sign of scores if needed...')
    # Flip signs of SSFs if needed
    if n_nc > 0:
        for i in range(len(eigVal)):
            mean_nc = np.mean(score_vectors[:n_nc, i])
            mean_disease = np.mean(score_vectors[n_nc:, i])
            if mean_nc > mean_disease:
                print(f'change sign of PC: {i}')
                score_vectors[:, i] *= -1

    print('Computing GIS...')
    # Compute voxel pattern eigenvectors or GIS
    gis = subject_residual_profiles.T @ score_vectors

    print('Z-transforming GIS...')
    # Z-transform the GIS
    for i in range(n_subj):
        mean = np.sum(gis[:, i]) / mask_sum
        std = np.std(gis[gis[:, i] != 0, i])
        gis[:, i] = (gis[:, i] - mean) / std

    # Mask the GIS
    gis = gis * mask_row[:, np.newaxis]

    if save_dir:
        np.save(vector_save_dir + f'/GIS.npy', gis)

    return gis, score_vectors, vaf, voxels_means


def pattern_biomarker_analysis(gis, score_vectors, vaf, filelist, labels, img_shape=(91,109,91), save_dir=None):
    
    all_score_dfs = []

    print('Analysing PC scores (NC and disease comparisons)...')
    for pc, vaf_pc in enumerate(vaf):
        if vaf_pc > 5:
            display_gis(gis, pc, img_shape, save_dir) # Display GIS in image format
            df = pc_scores_to_df(score_vectors, pc, filelist, labels) #Put current PC scores into a dataframe
            p = compare_scores_normal_disease(df, pc, labels, save_dir) #Plot comparisons between normal controls and diseased in this PC
            all_score_dfs.append(df)
    
    df_all_scores = pd.concat(all_score_dfs)

    all_pcs = list(df_all_scores['PC'].unique())

    print('Comparing all PC combinations...')
    # Logistic regression to find the best combination of PCs
    df_aic, df_vafs, df_p, df_all_scores, coefs = best_pc_combination(df_all_scores, vaf, save_dir)


    print('Combining PC vectors...')
    results = combine_pc_vectors(gis, coefs, save_dir)

    df_all_scores.to_csv(save_dir + '/scores.csv')

    
    if save_dir:
        df_aic.to_csv(save_dir + '/AIC.csv')
        df_p.to_csv(save_dir + '/p_values.csv')
        df_vafs.to_csv(save_dir + '/VAFs.csv')

    print('AIC:\n')
    print(df_aic.to_string())
    
    print('Vaf:\n')
    print(df_vafs.to_string())

    print('P-values:\n')
    print(df_p.to_string())




if __name__ == '__main__':

    # Parse config file path
    parser = ArgumentParser()
    parser.add_argument("--csv_file", type=str, help='CSV file with filepaths and corresponding labels of normal controls (0) or diseased subjects (1). Columns should be filepaths and labels.')
    parser.add_argument("--mask_file", help='Path to a binary brain mask file to apply to the images or threshold to create mask.')
    parser.add_argument("--image_shape", type=tuple, help='Image shape to reconstruct back the PC pattern images. Default is the MNI brain space dimensions.',
                        default=(91, 109, 91))
    parser.add_argument("--preprocess_img", type=bool, help='Whether to preprocess the image.')
    parser.add_argument("--save_dir", type=str, help='Directory path where to save the results files.')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_file)

    filelist = list(df['filepaths'])
    labels = list(df['labels'])


    gis, score_vectors, vaf, voxels_means = ssm_pca(filelist, labels,
                                    preprocess_img=args.preprocess_img,
                                    mask_file=args.mask_file,
                                    save_dir=args.save_dir)

    pattern_biomarker_analysis(gis, score_vectors, vaf, filelist, labels, img_shape=args.image_shape, 
                            save_dir=args.save_dir)
