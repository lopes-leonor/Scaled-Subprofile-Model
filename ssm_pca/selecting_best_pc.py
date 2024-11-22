
from itertools import permutations, combinations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score, RocCurveDisplay, log_loss
from sklearn.linear_model import LogisticRegression

from visualization import plot_comparisons
from voxel_pattern_analysis import get_scores_discovery_set, reshape_eigenvector_to_3d


def best_aic(df):
    pcs = list(df['PC'].unique())

    all_combinations = []
    for r in range(1, len(pcs) + 1):
        all_combinations.extend(combinations(range(len(pcs)), r))

    aics = dict()

    for combination in all_combinations:
        scores_comb = []
        for i in combination:
            df_i = df[df['PC'] == i]
            scores_i = list(df_i['norm_scores'])
            scores_comb.append(scores_i)

        scores = np.array(scores_comb).T
        labels = list(df_i['labels'])

        logistic_model = LogisticRegression()
        logistic_model.fit(scores, labels)
        aic = compute_aic(logistic_model, scores, labels)
        aics[combination] = aic

    df = pd.DataFrame([aics])

    return df


def comparisons_scores_normal_disease(df, save_dir):

    pcs = df['PC'].unique()

    for pc in pcs:
        df_temp = df[df['PC'] == pc]

        all_scores = df_temp['norm_scores'].to_numpy()
        labels = df_temp['labels'].to_numpy()

        normal = df_temp[df_temp['labels'] == 0]['scores']
        disease = df_temp[df_temp['labels'] == 1]['scores']

        #Do comparisons between normal controls and diseased
        _, p = ttest_ind(normal, disease)
        plot_comparisons(normal, disease, 'Controls', 'Disease', pc, save_dir=save_dir)

        auc = roc_auc_score(labels, all_scores)
        RocCurveDisplay.from_predictions(labels, all_scores, name=f'PC{pc} - AUC = {auc:3f}')
        plt.title(f'PC: {pc}')
        if save_dir:
            plt.savefig(save_dir + f'/ROC_PC{pc}.png')

        plt.plot()





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

# def evaluate_pcs_with_aic(score_vectors, labels, max_components=10):
#     """
#     Evaluate combinations of principal components to determine the model with the lowest AIC.
#
#     Parameters:
#     - score_vectors: The matrix of principal component scores
#     - labels: The target labels (patients vs. controls)
#     - max_components: The maximum number of components to consider (default is 10)
#
#     Returns:
#     - best_model: The logistic regression model with the lowest AIC
#     - best_components: The list of component indices for the best model
#     - best_aic: The AIC value of the best model
#     """
#     best_aic = np.inf
#     best_model = None
#     best_components = None
#
#     # Try different combinations of principal components
#     for num_components in range(1, max_components + 1):
#         for combination in itertools.combinations(range(score_vectors.shape[1]), num_components):
#             X = score_vectors[:, combination]
#
#             model = LogisticRegression().fit(X, labels)
#             aic = compute_aic(model, X, labels)
#
#             if aic < best_aic:
#                 best_aic = aic
#                 best_model = model
#                 best_components = combination
#
#
#     print(best_components)
#
#     return best_model, best_components, best_aic

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


def linearly_combine_vectors(normalized_gis, coeff_dict, combination):
    # Initialize the result as a zero vector with the same length as the number of rows in gis
    result_vector = np.zeros(normalized_gis.shape[0])

    # Linearly combine the normalized vectors with the current permutation of coefficients
    for pc in combination:
        result_vector += coeff_dict.get(pc) * normalized_gis[:, pc]

    return result_vector

def combine_pc_vectors(df, gis, save_dir):
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

        reshape_eigenvector_to_3d(result_vector, plot=False, save_name=f'PC_{comb_name}',
                                  save_dir=save_dir)

        results[comb_name] = result_vector

    return results


# if __name__ == '__main__':
#     a = ['a', 'b', 'c', 'd']
#
#     all_combinations = []
#     for r in range(1, len(a) + 1):
#         all_combinations.extend(combinations(a, r))
#
#     print(list(all_combinations))