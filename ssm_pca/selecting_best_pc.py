
from itertools import permutations, combinations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score, RocCurveDisplay, log_loss
from sklearn.linear_model import LogisticRegression

from ssm_pca.utils import reshape_eigenvector_to_3d, reshape_eigenvector_to_roi
from ssm_pca.visualization import plot_comparisons


def best_pc_combination(df, vaf=None, save_dir=None):

    df_temp = df.copy()

    pcs = list(df_temp['PC'].unique())

    # Get all possible combinations
    all_combinations = []
    for r in range(1, len(pcs) + 1):
        all_combinations.extend(combinations(range(len(pcs)), r))

    aics = dict()
    vafs_dict = dict()
    p_dict = dict()
    coefs_dict = dict()
    resulting_scores_dict = dict()

    labels = list(df_temp[df_temp['PC'] == 0]['labels'])
    filepaths = list(df_temp[df_temp['PC'] == 0]['filepaths'])

    # Iterate over the different combinations
    for combination in all_combinations:
        scores_comb = []
        vafs = []
    
        for i in combination:
            df_i = df_temp[df_temp['PC'] == i]
            scores_i = list(df_i['scores'])
            scores_comb.append(scores_i)
            vafs.append(vaf[i])

        scores = np.array(scores_comb).T

        # Fit logistic regression model
        logistic_model = LogisticRegression()
        logistic_model.fit(scores, labels)
        coefs = logistic_model.coef_.flatten()

        # Normalize coefficients
        coefs_normalized = coefs / np.sqrt(np.sum(coefs ** 2))
        coefs_dict[combination] = coefs_normalized

        # Get combination scores
        resulting_scores = scores @ coefs_normalized

        # Save resulting scores
        resulting_scores_dict[combination] = resulting_scores
        comb_list = [combination] * len(filepaths)
        df_resulting_scores = pd.DataFrame({'filepaths': filepaths, 'labels': labels, 'scores': resulting_scores, 'PC': comb_list})
        df_temp = pd.concat([df_temp, df_resulting_scores])

        # Compare current combination scores
        p = compare_scores_normal_disease(df_resulting_scores, combination, labels, save_dir)
        p_dict[combination] = p

        # Compute Vaf 
        tvaf = np.sum((coefs_normalized ** 2) * np.array(vafs))
        vafs_dict[combination] = tvaf

        # Compute AIC
        aic = compute_aic(logistic_model, scores, labels)
        aics[combination] = aic

        vafs.clear()
        scores_comb.clear()

    
    df_aics = pd.DataFrame([aics])
    df_vafs = pd.DataFrame([vafs_dict])
    df_p = pd.DataFrame([p_dict])

    return df_aics, df_vafs, df_p, df_temp, coefs_dict



def compare_scores_normal_disease(df, pc, labels, save_dir=None):
    
    all_scores = df['scores'].to_numpy()
    labels = np.array(labels)

    normal = df[df['labels'] == 0]['scores']
    disease = df[df['labels'] == 1]['scores']

    #Do comparisons between normal controls and diseased
    _, p = ttest_ind(normal, disease)
    plot_comparisons(normal, disease, 'Controls', 'Disease', pc, save_dir=save_dir + '/plots/')

    auc = roc_auc_score(labels, all_scores)
    RocCurveDisplay.from_predictions(labels, all_scores, name=f'PC{pc} - AUC = {auc:3f}')
    plt.title(f'PC: {pc}')

    if save_dir:
        plt.savefig(save_dir + f'plots/ROC_PC{pc}.png')

    plt.plot()
    plt.show(block=False)
    plt.close()

    return p


def compute_aic(model, X, y):
    """
    Compute the Akaike Information Criterion (AIC) for a given model.

    Args:
        model: The fitted logistic regression model
        X: The input features (principal components)
        y: The target labels (patients vs. controls)

    Returns:
        aic: The AIC value
    """
    # Number of parameters (including the intercept)
    k = X.shape[1] + 1

    # Log-likelihood of the model
    log_likelihood = -log_loss(y, model.predict_proba(X), normalize=False)

    # AIC computation
    aic = 2 * k - 2 * log_likelihood
    return aic


def normalize_vectors(vectors):
    """
    Normalize the selected vectors to have unit length.

    Args:
        vectors: 2D numpy array of selected vectors (n_voxels, n_selected_components)

    Returns:
        normalized_vectors: 2D numpy array of normalized vectors (n_voxels, n_selected_components)
    """
    norms = np.linalg.norm(vectors, axis=0)
    normalized_vectors = vectors / norms
    return normalized_vectors


def determine_coefficients(scores, labels):
    """
    Determine coefficients for linear combination using logistic regression.

    Args:
        scores: 1D numpy array of scores from one PC (n_samples, )
        labels: 1D numpy array of class labels (n_samples,)

    Returns:
        coefficient: regression coefficient
    """
    logistic_model = LogisticRegression()
    logistic_model.fit(scores, labels)
    coefficient = logistic_model.coef_.flatten()
    aic = compute_aic(logistic_model, scores, labels)

    return coefficient, aic


def linearly_combine_vectors(normalized_gis, coefs, combination):
    # Initialize the result as a zero vector with the same length as the number of rows in gis
    result_vector = np.zeros(normalized_gis.shape[0])

    # Linearly combine the normalized vectors with the current permutation of coefficients
    for i, pc in enumerate(combination):
        result_vector += coefs[i] * normalized_gis[:, pc]


    return result_vector

def combine_pc_vectors(gis, coeffs, save_dir):
    """
    Linearly combine the principal component vectors in GIS using logistic regression coefficients.

    Args:
        df: Dataframe with the scores of normal and disease subjects to fit logistic regression
        gis: 2D numpy array of component vectors (n_voxels, n_selected_components)
        save_dir: directory where to save the results

    Returns:
        combined_pattern: 1D numpy array of the combined pattern (n_voxels,)
    """


    # Normalize the gis matrix
    normalized_gis = normalize_vectors(gis)

    # Calculate the result for each combination
    results = {}

    all_combinations = coeffs.keys()

    for combination in all_combinations:
        if len(combination) > 1:
            
            result_vector = linearly_combine_vectors(normalized_gis, coeffs.get(combination), combination)
            comb_name = "_".join(map(lambda x: str(x), combination))

            np.save(save_dir + f'vectors/GIS_vector_PC_{comb_name}.npy', result_vector)

            reshape_eigenvector_to_3d(result_vector, plot=False, save_name=f'nifti/PC_{comb_name}',
                                    save_dir=save_dir)

            results[comb_name] = result_vector

    return results


def combine_pc_vectors_roi(df, gis, roi_mask_file, save_dir):
    """
    Linearly combine the principal component vectors in GIS using logistic regression coefficients.

    Args:
        df: Dataframe with the scores of normal and disease subjects to fit logistic regression
        gis: 2D numpy array of component vectors (n_voxels, n_selected_components)
        save_dir: directory where to save the results

    Returns:
        combined_pattern: 1D numpy array of the combined pattern (n_voxels,)
    """

    pcs = df['PC'].unique()

    coeffs = dict()
    aics = dict()

    # Get the logistic regression coefficients for the individual PCs
    for pc in pcs:
        df_temp = df[df['PC'] == pc]

        all_scores = df_temp['norm_scores'].to_numpy()
        all_scores = all_scores.reshape(-1, 1)

        labels = df_temp['labels']
        coef, aic = determine_coefficients(all_scores, labels)
        coeffs[pc] = coef
        aics[pc] = aic

    # Normalize the gis matrix
    normalized_gis = normalize_vectors(gis)

    # Generate all possible combinations of the indices of coefficients
    all_combinations = []
    for r in range(2, len(coeffs) + 1):
        all_combinations.extend(combinations(range(len(coeffs)), r))

    # Calculate the result for each combination
    results = {}

    for combination in all_combinations:
        result_vector = linearly_combine_vectors(normalized_gis, coeffs, combination)
        comb_name = "_".join(map(lambda x: str(x), combination))
        #np.save(save_dir + f'/GIS_vector_PC_{comb_name}.npy', result_vector)

        reshape_eigenvector_to_roi(result_vector, roi_mask_file=roi_mask_file, plot=True, save_name=f'PC_{comb_name}',
                                  save_dir=save_dir)

        results[comb_name] = result_vector

    return results

