
import numpy as np

def convert_to_row_vector(img):
    """Convert to continuous row vector"""
    row_vector = img.flatten()
    return row_vector

def create_group_matrix(row_vectors_list):
    """Stack all row vectors to form a group matrix"""
    matrix = np.vstack(row_vectors_list)
    return matrix

def transform_to_log(data, epsilon = 1e-4):
    """ Transform data to logarithmic form. Epsilon is added to deal with zero values. """
    log_data = np.log(data + epsilon)
    return log_data


def center_data_matrix(data_matrix):
    """
    Center the group matrix by subtracting row means.

    Args:
        data_matrix: Group data matrix, usually after logarithmic transformation.

    Returns:
        centered_data_matrix: Centered data matrix.
        group_mean_profile: Group mean profile (GMP).
    """

    row_means = np.mean(data_matrix, axis=1, keepdims=True)
    centered_data_matrix = data_matrix - row_means

    group_mean_profile = np.mean(centered_data_matrix, axis=0) # column means

    return centered_data_matrix, group_mean_profile


def compute_subject_residual_profiles(centered_data_matrix, group_mean_profile):
    """
    Compute subject residual profiles (SRP) by subtracting column means (or GMP) to row-centered data matrix.
    Args:
        centered_data_matrix: Group data matrix already centered for row means
        group_mean_profile: Group mean profile (GMP) or column means

    Returns:
        subject_residual_profiles: Matrix of subject residual profiles (SRP)
    """

    subject_residual_profiles = centered_data_matrix - group_mean_profile

    return subject_residual_profiles


def compute_subject_by_subject_covariance_matrix(subject_residual_profiles):
    """
    Compute the subject-by-subject covariance matrix C.

    Args:
        subject_residual_profiles: Matrix of subject residual profiles (SRP).

    Returns:
        covariance_matrix: Subject-by-subject covariance matrix C.
    """

    num_subjects = subject_residual_profiles.shape[0]
    covariance_matrix = np.zeros((num_subjects, num_subjects))

    for i in range(num_subjects):
        for j in range(num_subjects):
            covariance_matrix[i, j] = np.sum(subject_residual_profiles[i] * subject_residual_profiles[j])

    return covariance_matrix
