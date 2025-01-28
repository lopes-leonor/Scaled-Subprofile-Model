
import numpy as np
import nibabel as nb
import pandas as pd

from ssm_pca.visualization import plot_several_axes_image_array


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

def reshape_eigenvector_to_roi(voxel_pattern_vector, roi_mask_file, plot=False, save_name=None, save_dir=None):
    """
    Reshapes a roi principal component vector back into the original 3D roi image shape.

    Args:
        voxel_pattern_vector: 1D numpy array of voxel pattern vector corresponding to principal component index (pc) (num_rois, )
        roi_mask_file: path to the roi file
        save_name: name of the image to save
        save_dir: directory where to save 

    Returns
        reshaped_vector: 3D numpy array of the selected component in the original roi image shape
    """

    roi_mask = nb.load(roi_mask_file).get_fdata()
    reshaped_vector = np.zeros(roi_mask.shape)

    unique_labels = np.unique(roi_mask)[1:]  # Exclude background

    for region_idx, value in zip(unique_labels, voxel_pattern_vector):
        reshaped_vector[roi_mask == region_idx] = value

    if plot:
        plot_several_axes_image_array(reshaped_vector, rotations=(90,90,90), title=save_name)

  
    if save_dir:
        nb.save(nb.Nifti1Image(reshaped_vector, np.eye(4)), save_dir + '/' + save_name)
    
    return reshaped_vector

def display_gis(gis, pc, img_shape, save_dir=None):
    """Reshape the GIS vector into image format and plots it.

    Args:
        gis: GIS matrix.
        pc: Index of the GIS vector to display.
        save_dir: Directory to save the results. If None, results are not saved. Defaults to None.
    """
    save_name = f'PC_{pc}.nii'

    gis_pc = gis[:, pc]

    if save_dir:
        np.save(save_dir + f'vectors/GIS_vector_PC_{pc}.npy', gis_pc)

    reshape_eigenvector_to_3d(gis_pc, img_shape, plot=False, save_name=save_name, save_dir=save_dir + '/nifti/')

def pc_scores_to_df(score_vectors, pc, filepaths, labels):
    """Get the scores of the subjects for a given principal component.

    Args:
        score_vectors: Score vectors matrix.
        pc: Index of the principal component.
    """
    scores = score_vectors[:, pc]

    df = pd.DataFrame({'filepaths': filepaths, 'labels': labels, 'scores': scores, 'PC': pc})
    return df