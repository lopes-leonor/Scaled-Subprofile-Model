


from argparse import ArgumentParser
import numpy as np
import nibabel as nb
import pandas as pd

from ssm_pca.image_preprocess import preprocess
from ssm_pca.masking import mask_img
from ssm_pca.get_rois import get_rois
from ssm_pca.ssm_preprocess import convert_to_row_vector, create_group_matrix, transform_to_log, center_data_matrix, compute_subject_residual_profiles
from ssm_pca.pca import apply_pca, weight_score_vectors, compute_voxel_pattern_eigenvectors, calculate_vaf, plot_eigenvalues_vaf
from ssm_pca.voxel_pattern_analysis import z_transform_voxel_patterns, get_scores_discovery_set_roi, reshape_eigenvector_to_roi

from ssm_pca.selecting_best_pc import best_aic, combine_pc_vectors_roi, comparisons_scores_normal_disease


def ssm_pca(filelist=None, labels=None, img_shape=(91,109,91), preprocess_img=True, 
            mask_file=None, roi_mask_file=None, save_dir=None):

    """ Main function to run Scaled Subprofile Model with Principal Component Analysis on the images on filelist.

    Args:
        filelist: List with paths to the images to use. Defaults to None.
        labels: List to the labels corresponding to the images. Defaults to None.
        img_shape: Shape of images. Defaults to (91,109,91).
        preprocess_img: Whether to preprocess the images. Defaults to True.
        mask_file: Path to a mask file to mask the images. If None, no mask is applied Defaults to None.
        save_dir: Directory to save the results. If None, results are not saved. Defaults to None.

    Returns:
        z_transf_gis - z_transformed_patterns: 2D numpy array of Z-transformed voxel pattern vectors (n_voxels, n_components)  
        gis - voxel pattern eigenvectors
        vaf - variace accounted for by each component
        gmp - group mean profile
    """

    row_vectors_list = []

    for filepath in filelist:
        img = nb.load(filepath)
        array = img.get_fdata()

        # Preprocess image (change dimensions, spatial normalize, intensity normalize, etc)
        if preprocess_img:
            array = preprocess(array, ref_region_mask_file=mask_file)

        # Mask with given file
        if mask_file:
            array = mask_img(array, mask_file=mask_file)

        # Convert the array to a row vector
        #row_vector = convert_to_row_vector(array)

        roi_mask = nb.load(roi_mask_file).get_fdata()

        means, labels = get_rois(array, rois_mask=roi_mask)

        row_vector = means

        row_vectors_list.append(row_vector)

    # Create group matrix
    D = create_group_matrix(row_vectors_list)

    # Log transform group matrix
    log_data_matrix = transform_to_log(D)

    # Center the data matrix (subtract the row means). Get GMP - column means.
    centered_data_matrix, gmp = center_data_matrix(log_data_matrix)

    if save_dir:
        np.save(save_dir + '/group_mean_profile.npy', gmp)

    # Compute subject residual profiles (SRP) - subtract column means
    subject_residual_profiles = compute_subject_residual_profiles(centered_data_matrix, gmp)

    # Compute subject-by-subject covariance matrix
    covariance_matrix = np.dot(subject_residual_profiles, subject_residual_profiles.T)

    # Apply PCA to the covariance matrix
    score_vectors, eigenvectors, eigenvalues = apply_pca(covariance_matrix)

    # Calculate the variance accounted for (vaf) by each component
    vaf = calculate_vaf(eigenvalues)

    if save_dir:
        # Plot the variance accounted for (vaf) by each component
        plot_eigenvalues_vaf(eigenvalues, vaf)

    # Weight each score vector by the square root of its corresponding eigenvalue
    weighted_score_vectors = weight_score_vectors(eigenvalues, score_vectors)

    # Compute voxel pattern eigenvectors or GIS
    gis = compute_voxel_pattern_eigenvectors(subject_residual_profiles, weighted_score_vectors)
    z_transf_gis = z_transform_voxel_patterns(gis)

    if save_dir:
        np.save(save_dir + f'/GIS_matrix.npy', z_transf_gis)

    return z_transf_gis, gis, vaf, gmp


def display_save_individual_PC(z_transf_gis, gis, vaf, gmp, filelist, labels, mask_file=None, roi_mask_file=None, save_dir=None):

    all_score_dfs = []

    # Display and save the GIS of each PC with Vaf > 5%
    for pc, vaf_pc in enumerate(vaf):
        if vaf_pc > 5:
            z_transf_gis_pc = z_transf_gis[:, pc]
            if save_dir:
                np.save(save_dir + f'/GIS_vector_PC{pc}.npy', z_transf_gis_pc)

            df = get_scores_discovery_set_roi(filelist, labels, mask_file, roi_mask_file, z_transf_gis_pc, pc, gmp, save_dir)
            all_score_dfs.append(df)
            save_name = f'PC_{pc}.nii'

            reshape_eigenvector_to_roi(z_transf_gis_pc, roi_mask_file, plot=True, save_name=save_name, save_dir=save_dir)
            #reshape_eigenvector_to_3d(z_transf_gis_pc, img_shape, plot=False, save_name=save_name, save_dir=save_dir)


    df_final_individual = pd.concat(all_score_dfs)
    if save_dir:
        df_final_individual.to_csv(save_dir + '/scores_individual_PC.csv')

    df_aic = best_aic(df_final_individual)

    if save_dir:
        df_aic.to_csv(save_dir + '/AIC.csv')
    print(df_aic.to_string())

    # Get dictionary of combination (ex: 1_2_3) to corresponding gis vector
    combined_gis = combine_pc_vectors_roi(df_final_individual, z_transf_gis, roi_mask_file, save_dir)

    for pc_comb, gis_pc in combined_gis.items():
        pc_comb_name = f'PC_{pc_comb}'
        df = get_scores_discovery_set_roi(filelist, labels, mask_file, roi_mask_file, gis_pc, pc_comb_name, gmp, save_dir)
        all_score_dfs.append(df)

    df_final = pd.concat(all_score_dfs)

    df_final.to_csv(save_dir + '/scores.csv')

    comparisons_scores_normal_disease(df_final, save_dir)



if __name__ == '__main__':

    # Parse config file path
    parser = ArgumentParser()
    parser.add_argument("--csv_file", type=str, help='CSV file with filepaths and corresponding labels of normal controls (0) or diseased subjects (1). Columns should be filepaths and labels.')
    parser.add_argument("--mask_file", type=str, help='Path to a binary brain mask file to apply to the images.')
    parser.add_argument("--roi_mask_file", type=str, help='Path to a brain ROI mask file to apply to the images.')
    parser.add_argument("--image_shape", type=tuple, help='Image shape to reconstruct back the PC pattern images. Default is the MNI brain space dimensions.',
                        default=(91, 109, 91))
    parser.add_argument("--preprocess_img", type=bool, help='Whether to preprocess the image.')
    parser.add_argument("--save_dir", type=str, help='Directory path where to save the results files.')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_file)

    filelist = list(df['filepaths'])
    labels = list(df['labels'])

    
    z_transf_gis, gis, vaf, gmp = ssm_pca(filelist=args.filelist, labels=args.labels,
                                           mask_file=args.mask_file, roi_mask_file=args.roi_mask_file, img_shape=args.image_shape,
                                           preprocess_img=args.preprocess_img, save_dir=args.save_dir)


    display_save_individual_PC(z_transf_gis, gis, vaf, gmp, filelist, labels, mask_file=args.mask_file, roi_mask_file=args.roi_mask_file,
                                      save_dir=args.save_dir)
    




