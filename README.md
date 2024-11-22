# SSM-PCA

by Leonor Lopes


Implementation based on the following papers: 

Spetsieris, P. G., & Eidelberg, D. (2011). Scaled subprofile modeling of resting state imaging data in Parkinson's disease: methodological issues. NeuroImage, 54(4), 2899–2914. https://doi.org/10.1016/j.neuroimage.2010.10.025

Spetsieris, P., Ma, Y., Peng, S., Ko, J. H., Dhawan, V., Tang, C. C., & Eidelberg, D. (2013). Identification of disease-related spatial covariance patterns using neuroimaging data. Journal of visualized experiments : JoVE, (76), 50319. https://doi.org/10.3791/50319

----- 

<p>

## Introduction

Code to perform Scaled Subprofile Model (SSM), a principal components analysis (PCA)-based spatial covariance method.

Spatial covariance patterns are derived as PCA eigenvectors reflecting significant sources of variance in the data. These patterns are also called Group Invariant Subprofiles - GISs.

They can represent:
- neuroanatomical/functional brain networks - interconected brain regions
- involvement of brain regions in a given clinical/neuropathological condition


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

## Contributions
Contributions to improve preprocessing steps, analysis, or visualization are welcome. Please fork the repository and submit a pull request.

## Licence
This package is licensed under the MIT License. See the LICENSE file for details.
