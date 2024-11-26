# SSM-PCA

by Leonor Lopes


Implementation based on the following papers: 

1. Spetsieris, P. G., & Eidelberg, D. (2011). Scaled subprofile modeling of resting state imaging data in Parkinson's disease: methodological issues. NeuroImage, 54(4), 2899–2914. https://doi.org/10.1016/j.neuroimage.2010.10.025

2. Spetsieris, P., Ma, Y., Peng, S., Ko, J. H., Dhawan, V., Tang, C. C., & Eidelberg, D. (2013). Identification of disease-related spatial covariance patterns using neuroimaging data. Journal of visualized experiments : JoVE, (76), 50319. https://doi.org/10.3791/50319

----- 

<p>

## Introduction

Code to perform Scaled Subprofile Model (SSM), a principal components analysis (PCA)-based spatial covariance method.

Spatial covariance patterns are derived as PCA eigenvectors reflecting significant sources of variance in the data. These patterns are also called Group Invariant Subprofiles - GISs.

They can represent:
- neuroanatomical/functional brain networks - interconected brain regions
- involvement of brain regions in a given clinical/neuropathological condition

<img src="https://github.com/lopes-leonor/Scaled-Subprofile-Model/blob/main/images/PC_0.png" width="800" alt="Parkinson's Disease Related Pattern">
Parkinson's Disease Related Pattern<br>
<br>


## Installation

The package is available on PyPI. You can install it with:

```bash
pip install ssm-pca
```

## Usage

Using the Command Line:

```bash
python -m ssm_pca --csv_file "data.csv" --mask_file "brain_mask.nii" --save_dir "results/" --image_shape "(91,109,91)" --preprocess_img True

```

Using Python Code:

```bash

from ssm_pca import ssm_pca, display_save_individual_PC

# Define file paths and labels
filelist = ["control.nii", "disease.nii"]
labels = [0, 1]


# Apply SSM-PCA in list of images and labels   
z_transf_gis, gis, vaf, gmp = ssm_pca(
    filelist = filelist, 
    labels = labels,
    mask_file = "brain_mask.nii", 
    img_shape = (91, 109, 91),
    preprocess_img = True, 
    save_dir = "results/")

# Display and save the GIS - spatial covariance patterns - corresponding to each PC (with Vaf > 5%)
display_save_individual_PC(
    z_transf_gis, gis, vaf, gmp, 
    mask_file = "brain_mask.nii",
    save_dir = "results/")

```


| Parameter         | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| `--csv_file`      | CSV file with filepaths and corresponding labels of normal controls (0) or diseased subjects (1). Columns should be filepaths and labels.                 |
| `--mask_file`     | Path to a binary mask file to apply to the images.                |
| `--save_dir`      | Directory to save the results.                                    |
| `--image_shape`   | Image shape to reconstruct back the PC pattern images. Default is the MNI brain space dimensions                      |
| `--preprocess_img`| Whether to preprocess the images - intensity normalization (True or False).                 |


## Contributions
Contributions to improve preprocessing steps, analysis, or visualization are welcome. Please fork the repository and submit a pull request.


## Licence
This package is licensed under the MIT License. See the LICENSE file for details.
