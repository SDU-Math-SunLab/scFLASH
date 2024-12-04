from sklearn.metrics import f1_score
import numpy as np  
import scanpy as sc


def calculate_knn_performance(adata, condition='condition'):
    """
    Calculate the performance of a KNN classifier.

    Parameters:
        adata (AnnData): AnnData object containing cell connectivity information and condition labels.
        condition_key (str): Key name for the condition labels. Defaults to 'condition'.

    Returns:
        f1_micro (float): The micro-averaged F1 score of the KNN classifier.
    """
    # Retrieve neighbor indices from the connectivity matrix
    connectivities = adata.obsp['connectivities'].toarray()

    # Retrieve condition values
    conditions = adata.obs[condition]

    most_knn_conditions = []
    # Determine the most common condition among each cell's neighbors
    for i in range(connectivities.shape[0]):
        neighbor_indices = connectivities[i].nonzero()[0]
        neighbor_conditions = conditions[neighbor_indices]
        unique_conditions, counts = np.unique(neighbor_conditions, return_counts=True)
        most_common_condition = unique_conditions[np.argmax(counts)]
        most_knn_conditions.append(most_common_condition)

    # Calculate the F1 score
    f1_micro = f1_score(conditions, most_knn_conditions, average='micro')

    return f1_micro
