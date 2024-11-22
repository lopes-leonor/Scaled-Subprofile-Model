
import numpy as np
import nibabel as nb
import pandas as pd

from visualization import plot_several_axes_image_array
from prospective_score import tpr


def z_transform_voxel_patterns(voxel_patterns):
    """
    Z-transforms voxel pattern vectors so that their values represent positive and negative standard deviations from their mean value.

    Args:
        voxel_patterns: 2D numpy array of voxel pattern vectors (n_voxels, n_components)

    Returns:
        z_transformed_patterns: 2D numpy array of Z-transformed voxel pattern vectors (n_voxels, n_components)
    """

    mean_values = np.mean(voxel_patterns, axis=0)
    std_values = np.std(voxel_patterns, axis=0)
    z_transformed_patterns = (voxel_patterns - mean_values) / std_values

    return z_transformed_patterns

def get_scores_discovery_set(filelist, labels, mask_file, gis_pc, pc, group_mean_profile, save_dir):

    scores = []

    for filepath, label in zip(filelist, labels):
        _, score = tpr(filepath, mask_file, gis_pc, group_mean_profile)
        scores.append(score)


    # Z-score transformation
    z_scores = (np.array(scores) - np.mean(scores)) / np.std(scores)

    data = {'filepaths': filelist, 'scores': scores, 'z_scores': list(z_scores), 'labels': labels, 'PC': pc}
    df = pd.DataFrame(data)

    mean_nc = df[df['labels'] == 0]['z_scores'].mean()

    np.save(save_dir + '/mean_nc.npy', mean_nc)

    # Offset by normal controls mean
    df['norm_scores'] = df['z_scores'] - mean_nc


    if save_dir:
        # Save the scores into a dataframe
        df.to_csv(save_dir + f'/scores_PC_{pc}.csv')

    return df


def reshape_eigenvector_to_3d(voxel_pattern_vector, image_shape=(91, 109, 91), plot=False, save_name=None, save_dir=None):
    """
    Reshapes a specific principal component vector back into the original 3D image shape.

    Args:
        voxel_pattern_vector: 1D numpy array of voxel pattern vector corresponding to principal component index (pc) (num_voxels, )
        pc: Corresponding principal component index
        image_shape: tuple specifying the original 3D image shape (x, y, z)
        plot: Whether to display the 3D image of the pattern vector
        save_name: name of the image to save
        save_dir: directory where to save the 3D image

    Returns
        reshaped_vector: 3D numpy array of the selected component in the original image shape
    """

    reshaped_vector = voxel_pattern_vector.reshape(image_shape)

    # Display images of the GIS
    if plot:
        plot_several_axes_image_array(reshaped_vector, rotations=(90,90,90))

    if save_dir:
        nb.save(nb.Nifti1Image(reshaped_vector, np.eye(4)), save_dir + '/' + save_name)
