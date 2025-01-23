
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import ttest_ind
import nibabel as nib
from matplotlib.colors import Normalize, to_rgba



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
        np.save(save_dir + f'vectors/GIS_PC{pc}.npy', gis_pc)

    reshape_eigenvector_to_3d(gis_pc, img_shape, plot=False, save_name=save_name, save_dir=save_dir + '/nifti/')


def plot_comparisons(data1, data2, label1, label2, pc, save_dir=None):
    # Perform t-test
    t_stat, p_value = ttest_ind(data1, data2)

    #plt.ion()
    # Create boxplot
    plt.figure(figsize=(10, 6))

    df = pd.DataFrame({'labels': [label1]*len(data1) + [label2]*len(data2),
                       'scores': np.concatenate([data1, data2])})

    #df = pd.DataFrame({label1: data1, label2: data2})

    sns.boxplot(data=df, x='labels', y='scores')
    sns.swarmplot(df, x='labels', y='scores', color='black', alpha=0.5)

    plt.xticks([0, 1], [label1, label2])

    # Draw line and annotate p-value
    x1, x2 = 0, 1
    #y, h, col = max(max(data1), max(data2)), 1, 'k'
    plt.text(0.5, np.max(data1), f"p = {p_value:.3f}", ha='center', va='bottom', color='k')
    plt.title(f'PC: {pc}')
    if save_dir:
        plt.savefig(save_dir + f'/box_plot_scores_PC{pc}.png')
    # Show plot
    plt.show(block=False)
    #plt.close()

def plot_slice_image_array(array, slice_idx, title=None):

    plt.ion()
    # Choose a slice to visualize (you can change this as needed)
    slice_data = array[:,:,slice_idx]

    # Display the slice using Matplotlib'notebooks imshow
    plt.imshow(slice_data, cmap='jet')
    plt.colorbar()  # Add a color bar to show intensity scale
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plot_several_axes_image_array(array, slice_idxs:tuple=None, rotations:tuple=None, cmap=None, title:str=None, save_name:str=None):
    """
    Plot one slice per axis in a 3D image.
    Args:
        array: numpy array of the image
        slice_idxs: choose slice index to plot. Default is middle slice.
        rotations: angles to rotate the slices. Direction contrary to watch.
    """

    if len(array.shape) != 3:
        raise ValueError(f'Input image data ({title}) must be 3D')

    #plt.ion()
    # Create subplots for different views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    # Get min and max of array
    vmin = np.min(array)
    vmax = np.max(array)

    if slice_idxs == None:
        xy_idx = array.shape[2] // 2
        xz_idx = array.shape[1] // 2
        yz_idx = array.shape[0] // 2

        slice_idxs = [xy_idx, xz_idx, yz_idx]

    # Choose slice indices for different views
    slice_xy = array[:, :, slice_idxs[0]]
    slice_xz = array[:, slice_idxs[1], :]
    slice_yz = array[slice_idxs[2], :, :]

    # Rotate slices if needed
    if rotations:
        slice_xy = scipy.ndimage.rotate(slice_xy, rotations[0])
        slice_xz = scipy.ndimage.rotate(slice_xz, rotations[1])
        slice_yz = scipy.ndimage.rotate(slice_yz, rotations[2])


    # Plot slices for each view
    im0 = axes[0].imshow(slice_xy, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('XY Plane')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(slice_xz, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title('XZ Plane')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(slice_yz, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title('YZ Plane')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.colorbar(im0, cax=plt.axes([0.02, 0.12, 0.02, 0.7]), anchor=(vmin, vmax))

    if save_name:
        plt.savefig(save_name)
    plt.show(block=False)

def plot_mri_pet(mri, pet, slice_idxs:tuple=None, rotations:tuple=None, cmap='jet', title:str=None):


    if len(pet.shape) != 3:
        raise ValueError(f'Input image data ({title}) must be 3D')

    # Create subplots for different views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    # Get min and max of array
    vmin = np.min(pet)
    vmax = np.max(pet)

    if slice_idxs == None:
        xy_idx = pet.shape[2] // 2
        xz_idx = pet.shape[1] // 2
        yz_idx = pet.shape[0] // 2

        slice_idxs = [xy_idx, xz_idx, yz_idx]

    # Choose slice indices for different views
    slice_xy = pet[:, :, slice_idxs[0]]
    mask_xy = np.where(slice_xy == 0, 0, 0).astype(float)
    slice_xy_mri = mri[:, :, slice_idxs[0]]

    slice_xz = pet[:, slice_idxs[1], :]
    mask_xz = np.where(slice_xz == 0, 0, 0).astype(float)
    slice_xz_mri = mri[:, slice_idxs[1], :]

    slice_yz = pet[slice_idxs[2], :, :]
    mask_yz = np.where(slice_yz == 0, 0, 0).astype(float)
    slice_yz_mri = mri[slice_idxs[2], :, :]

    # Rotate slices if needed
    if rotations:
        slice_xy = scipy.ndimage.rotate(slice_xy, rotations[0])
        slice_xz = scipy.ndimage.rotate(slice_xz, rotations[1])
        slice_yz = scipy.ndimage.rotate(slice_yz, rotations[2])

        slice_xy_mri = scipy.ndimage.rotate(slice_xy_mri, rotations[0])
        slice_xz_mri = scipy.ndimage.rotate(slice_xz_mri, rotations[1])
        slice_yz_mri = scipy.ndimage.rotate(slice_yz_mri, rotations[2])


    # Plot slices for each view
    axes[0].imshow(slice_xy_mri, cmap='Greys', vmin=vmin, vmax=vmax, alpha=1)
    im0 = axes[0].imshow(slice_xy, cmap=cmap, vmin=vmin, vmax=vmax, alpha=mask_xy.astype(float))
    axes[0].set_title('XY Plane')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(slice_xz_mri, cmap='Greys', vmin=vmin, vmax=vmax)
    axes[1].imshow(slice_xz, cmap=cmap, vmin=vmin, vmax=vmax, alpha=mask_xz.astype(float))
    axes[1].set_title('XZ Plane')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(slice_yz_mri, cmap='Greys', vmin=vmin, vmax=vmax)
    axes[2].imshow(slice_yz, cmap=cmap, vmin=vmin, vmax=vmax, alpha=mask_yz.astype(float))
    axes[2].set_title('YZ Plane')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.colorbar(im0, cax=plt.axes([0.02, 0.12, 0.02, 0.7]), anchor=(vmin, vmax))
    plt.tight_layout()
    plt.show()

def plot_2_img(pet, mri, cmap1='viridis', cmap2='viridis', norm1=None, norm2=None):

    # Create subplots for different views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    if slice_idxs == None:
        xy_idx = pet.shape[2] // 2
        xz_idx = pet.shape[1] // 2
        yz_idx = pet.shape[0] // 2

        slice_idxs = [xy_idx, xz_idx, yz_idx]

    # Choose slice indices for different views
    slice_xy = pet[:, :, slice_idxs[0]]
    slice_xy_mri = mri[:, :, slice_idxs[0]]

    slice_xz = pet[:, slice_idxs[1], :]
    slice_xz_mri = mri[:, slice_idxs[1], :]

    slice_yz = pet[slice_idxs[2], :, :]
    slice_yz_mri = mri[slice_idxs[2], :, :]


    norm1 = Normalize(vmin=image1.min(), vmax=image1.max())
    norm2 = Normalize(vmin=image2.min(), vmax=image2.max())

    # Create RGB combined image
    combined_image = np.zeros((*image1.shape, 4))  # RGBA format
    mask2 = image2 > 0  # Mask for image2

    # Apply colormaps
    combined_image[~mask2] = to_rgba(cmap1(norm1(image1[~mask2])))  # Image1 where image2 == 0
    combined_image[mask2] = to_rgba(cmap2(norm2(image2[mask2])))    # Image2 where image2 > 0

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.title("Combined Image with Two Colormaps")
    plt.imshow(combined_image)
    plt.axis("off")
    plt.show()

if __name__ == '__main__':

    array = nib.load('/home/leonor/Code/ssm_pca/results/run4_186_definite_pd_186_hc/PC_0.nii').get_fdata()
    mri = nib.load('/home/leonor/Code/masks/SPM_masks/T1.nii').get_fdata()


    #array = np.where((array > 1) | (array < -1), array, 0)


    plot_several_axes_image_array(array, cmap='jet', rotations=(90, 90, 90), slice_idxs=(32, 53, 45))

    #plot_mri_pet(mri, array)
