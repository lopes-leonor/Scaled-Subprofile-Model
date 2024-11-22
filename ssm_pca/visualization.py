
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import ttest_ind
import nibabel as nib


def plot_comparisons(data1, data2, label1, label2, pc, save_dir=None):
    # Perform t-test
    t_stat, p_value = ttest_ind(data1, data2)

    # Create boxplot
    plt.figure(figsize=(10, 6))

    sns.boxplot(data=[data1, data2])
    #plt.swarm(data1, data2, color='k')

    plt.xticks([0, 1], [label1, label2])

    # Draw line and annotate p-value
    x1, x2 = 0, 1
    #y, h, col = max(max(data1), max(data2)), 1, 'k'
    plt.text((x1 + x2) * .5, np.max(data1), f"p = {p_value:.3f}", ha='center', va='bottom', color='k')
    plt.title(f'PC: {pc}')
    if save_dir:
        plt.savefig(save_dir + f'/box_plot_comparisons_PC{pc}.png')
    # Show plot
    plt.show()

def plot_slice_image_array(array, slice_idx, title=None):

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
    plt.show()

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

if __name__ == '__main__':

    array = nib.load('/home/leonor/Code/ssm_pca/results/run4_186_definite_pd_186_hc/PC_0.nii').get_fdata()
    mri = nib.load('/home/leonor/Code/masks/SPM_masks/T1.nii').get_fdata()


    #array = np.where((array > 1) | (array < -1), array, 0)


    plot_several_axes_image_array(array, cmap='jet', rotations=(90, 90, 90), slice_idxs=(32, 53, 45))

    #plot_mri_pet(mri, array)
