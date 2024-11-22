import glob
import os
import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score, RocCurveDisplay

from masking import mask_img
from ssm_preprocess import transform_to_log
from visualization import plot_comparisons


def tpr(filepath, mask_file, gis_pc, group_mean_profile, to_mni_space=True):
    """

    Returns:
        object:
    """
    img = nb.load(filepath)
    array = img.get_fdata()

    if to_mni_space:
        array = np.pad(array, ((6, 6), (7, 7), (11, 11)))

    array = mask_img(array, mask_file=mask_file)
    array = array / np.mean(array)
    # array = (array - np.mean(array)) / np.std(array)

    # Log transform group matrix
    array = transform_to_log(array)

    subject_mean = np.mean(array)

    vector = array.flatten()

    srp = vector - subject_mean - group_mean_profile

    score = np.dot(srp, gis_pc)

    return srp, score


def prospective_score(filelist, mask_file, gis_pc_file, group_mean_profile_file, mean_normals=None, to_mni_space=True, save_dir=None):

    gis_pc = np.load(gis_pc_file)
    gmp = np.load(group_mean_profile_file)
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

    df = pd.read_excel('/home/leonor/Code/RBD_model_comparison/data/huashan_RBD_FDG_DAT_data_info_2024.xlsx')
    #data_dir = '/home/leonor/Data/RBD_project/rbd_data/sn_spm/'

    # Exclude some patients

    df = df[df['no scan available_fdg'] == 0]
    #df = df[df['scans after conversion'] == 0]

    gis_pc = np.load('/home/leonor/Code/ssm_pca/results/run4_186_definite_pd_186_hc/GIS_vector_PC0.npy')
    group_mean_profile = np.load('/home/leonor/Code/ssm_pca/results/run4_186_definite_pd_186_hc/group_mean_profile.npy')
    scores = []

    for filepath in df['img_paths_fdg']:

        if filepath is not None:
            #file = data_dir + os.path.basename(filepath)

            srp, score = tpr(filepath, mask_file='/home/leonor/Code/masks/SPM_masks/grey_binary_prob25perc.nii',
                        gis_pc=gis_pc , group_mean_profile=group_mean_profile, to_mni_space=False)
        else:
            score = ''

        scores.append(score)

    # Z-score transformation
    scores_z = (np.array(scores) - np.mean(scores)) / np.std(scores)

    mean_normals = np.load('/home/leonor/Code/ssm_pca/results/run4_186_definite_pd_186_hc/mean_nc.npy')
    print(mean_normals)

    scores_z = scores_z - mean_normals

    df['PC_scores'] = scores_z

    df.to_excel('/home/leonor/Code/ssm_pca/results/run4_186_definite_pd_186_hc/RBD_results_run4_PC0.xlsx')

