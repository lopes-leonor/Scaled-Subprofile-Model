import os.path

import pandas as pd
import numpy as np
from sklearn.utils import resample
from ssm_pca.main_old import ssm_pca
from ssm_pca.visualization import plot_mri_pet, plot_several_axes_image_array
from tqdm import tqdm
import glob
import nibabel as nib



def calculate_icv(images):
    """
    Calculate the inverse coefficient of variation (ICV) for each voxel.
    ICV is defined as mean/standard deviation.
    """
    mean = np.mean(images, axis=0)
    std_dev = np.std(images, axis=0)

    # Avoid division by zero by using np.where
    icv = np.where(std_dev != 0, std_dev, 0)
    return icv

def create_mask(icv, threshold):
    """
    Create a mask of voxels where the coefficient of variation (CV) is greater than the threshold.
    """
    #mask = cv > threshold
    mask = np.where(icv > threshold, icv, 0)

    return mask

def bootstrapping(num_bootstraps, df_nc, df_pd, save_dir):
    # Perform bootstrapping
    for i in tqdm(range(num_bootstraps)):
        # Resample subjects (rows of the DataFrame) with replacement
        resampled_df_nc = resample(df_nc)
        resampled_df_pd = resample(df_pd)

        filelist_nc = list(resampled_df_nc['img_paths'])
        filelist_pd = list(resampled_df_pd['img_paths'])

        labels_nc = [0] * len(filelist_nc)
        labels_pd = [1] * len(filelist_pd)

        filelist = filelist_nc + filelist_pd
        labels = labels_nc + labels_pd

        z_transf_gis, gis, vaf, gmp = ssm_pca(filelist, labels, img_shape=(91, 109, 91),
                                              mask_file='/home/leonor/Code/masks/SPM_masks/grey_binary_prob25perc.nii',
                                              save_dir=None)

        z_transf_gis_pc = z_transf_gis[:, 0]  # Select the first PC

        reshaped_vector = z_transf_gis_pc.reshape((91, 109, 91))

        if save_dir:
            # Save each bootstrap result to disk immediately
            np.save(os.path.join(save_dir, f'bootstrapp_PC0_{i}.npy'), reshaped_vector)

def get_icv_image(save_dir):
    filelist = glob.glob(save_dir + '/bootstrapp_PC0_*.npy')
    z_score_maps = []

    for file in filelist:
        img = np.load(file)
        z_score_maps.append(img)

    images = np.array(z_score_maps)
    print(images.shape)

    icv = calculate_icv(images)

    nifti_img = nib.Nifti1Image(icv, affine=np.eye(4))  # affine is the transformation matrix, usually identity

    # Save the NIfTI image
    nib.save(nifti_img, '/home/leonor/Code/ssm_pca/results/run4_186_definite_pd_186_hc/icv_bootstrap.nii')

    mask = create_mask(icv, threshold=1.96)
    #print(mask)
    print(np.max(mask))

    x_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    for x in x_list:

        plot_several_axes_image_array(mask, slice_idxs= (x,x,50),
                                      save_name='/home/leonor/Code/ssm_pca/results/run4_186_definite_pd_186_hc/bootstrapp_PC0_icv_1_96.png')


if __name__ == '__main__':

    df_nc = pd.read_csv('/home/leonor/Code/ssm_pca/results/run5_186_definite_pd_186_hc/normal_controls_for_ssm.csv')
    df_pd = pd.read_csv('/home/leonor/Code/ssm_pca/results/run5_186_definite_pd_186_hc/pd_for_ssm.csv')

    save_dir = '/home/leonor/Code/ssm_pca/results/run5_186_definite_pd_186_hc/bootstrapping/'
    # Set the number of bootstraps
    num_bootstraps = 100

    bootstrapping(num_bootstraps, df_nc, df_pd, save_dir=save_dir)

    #get_icv_image(save_dir)

