
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.linalg as la

def apply_pca(covariance_matrix):
    """
    Apply PCA to the subject-by-subject covariance matrix C.

    Args:
        covariance_matrix: Subject-by-subject covariance matrix C.

    Returns:
        score_vectors: Projections of the original data points into the eigenvector space.
        eigenvalues: Eigenvalues of the covariance matrix.
        eigenvectors: Eigenvectors of the covariance matrix.
    """

    eigVal, eigVec = la.eigh(covariance_matrix)

    eigVal = np.real(eigVal)
    eigVec = np.real(eigVec)
    
    # Calculate the variance accounted for (vaf) by each component
    vaf = (eigVal / np.sum(eigVal))*100

    # Weight each score vector by the square root of its corresponding eigenvalue
    sqrt_eigVal = np.sqrt(eigVal)
    sqrt_eigVal = np.tile(sqrt_eigVal, (eigVec.shape[0], 1))

    score_vectors = eigVec * sqrt_eigVal 
    # score_vectors is a matrix of (subjects x GIS)
    # Each column represents a weighted eigenvector (dominant pattern of variability across subjects). 
    # Each element of the vector represents how the subject contributes or aligns with this pattern.
  
    # Reverse the order of eigenvalues, eigenvectors, and percent variance
    score_vectors = np.fliplr(score_vectors)
    vaf = np.flip(vaf)  # Percent variance accounted for
    eigVal = np.flip(eigVal)  # Eigenvalues


    return score_vectors, eigVal, vaf

def plot_eigenvalues_vaf(eigenvalues, vaf):

    fig, (ax1, ax2) = plt.subplots(2, 1)

    components = np.arange(1, len(vaf) + 1)

    ax1.plot(components, eigenvalues, color='skyblue', marker='o')
    #ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_xlim([0,20])

    # Disable x-axis ticks for the top plot
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax2.plot(components, vaf, color='skyblue', marker='o')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('% Vaf')
    ax2.set_xlim([0, 20])
    ax2.set_xticks(components)

    fig.suptitle('Eigenvalues and % Vaf by Each Principal Component')
    #plt.xticks(components)

    plt.grid(False)
    plt.show(block=False)
    #plt.close()


# def weight_score_vectors(eigenvalues, score_vectors):
#     """
#     Weight each vector by multiplying by the square root of its corresponding eigenvalue.

#     Args:
#         eigenvalues: Eigenvalues of the subject-by-subject covariance matrix.
#         score_vectors: Subject score eigenvectors. Projections of the original data points into the eigenvector space.

#     Returns:
#         weighted_score_vectors
#     """

#     weighted_score_vectors = score_vectors * np.sqrt(eigenvalues)
#     return weighted_score_vectors

# def compute_voxel_pattern_eigenvectors(subject_residual_profiles, weighted_score_vectors):
#     """
#     Compute voxel pattern eigenvectors using subject residual profiles (SRP) and PCA results.
#     Each column vector represents a principal component (PC) image pattern of the SSM/PCA analysis.

#     Args:
#         subject_residual_profiles: Matrix of subject residual profiles (SRP).
#         weighted_score_vectors: Weighted subject score eigenvectors.

#     Returns:
#         voxel_pattern_eigenvectors: Voxel pattern eigenvectors, corresponding to principal component (PC) image patterns
#         or GIS
#     """
#     voxel_pattern_eigenvectors = np.dot(subject_residual_profiles.T, weighted_score_vectors)

#     return voxel_pattern_eigenvectors


# def calculate_vaf(eigenvalues):
#     """
#     Calculate the percent variance accounted for by each principal component (PC).
#     Args:
#         eigenvalues: Eigenvalues of the subject-by-subject covariance matrix.

#     Returns:
#         vaf: percent variance accounted for

#     """
#     total_variance = np.sum(eigenvalues)
#     vaf = 100 * (eigenvalues / total_variance)
#     return vaf

