import random
import numpy as np
import torch
import scanpy as sc
import pandas as pd


def set_random_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    
def select_device(GPU = True):
    if GPU:
        if torch.cuda.is_available():
            if isinstance(GPU, str):
                device = torch.device(GPU)
            else:
                device = torch.device('cuda:0')
        else:
            print("GPU isn't available, and use CPU to train Docs.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def preprocess(adata, filter=True,n_genes=2000):
    # clear the obs &var names
    adata = adata[:, adata.var_names.notnull()]
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    
    
    if filter:
        # filter cells and genes
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
    
    adata.raw = adata.copy()
    # normalization
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_genes)
    adata = adata[:, adata.var.highly_variable]
    
    sc.pp.scale(adata)
    # print(adata)
    return adata


def factor(step):
    # Define some hyperparameters
    max_factor = 1.0 # The maximeanm value of the factor
    min_factor = 0.0 # The minimum value of the factor
    scale = 20.0 # The scale of the sigmoid function
    shift = 100.0 # The shift of the sigmoid function

    # Compute the factor using sigmoid function
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    return min_factor + (max_factor - min_factor) * sigmoid((step - shift) / scale)


def print_dataset_information(adata, batch_key="BATCH", celltype_key="celltype", log=None):
    if batch_key is not None and batch_key not in adata.obs.columns:
        print('Please check whether there is a {} column in adata.obs to identify batch information!'.format(batch_key))
        raise IOError

    if celltype_key is not None and celltype_key not in adata.obs.columns:  # 修正检查celltype_key的条件
        print('Please check whether there is a {} column in adata.obs to identify celltype information!'.format(celltype_key))
        raise IOError

    if log is not None:
        log.info("===========print brief information of dataset ===============")
        log.info("===========there are {} {}s in this dataset==============".format(len(adata.obs[batch_key].value_counts()), batch_key.lower()))
        log.info("===========there are {} {}s with this dataset=========".format(len(adata.obs[celltype_key].value_counts()), celltype_key.lower()))
    else:
        print("===========print brief information of dataset ===============")
        print("===========there are {} {}s in this dataset==============".format(len(adata.obs[batch_key].value_counts()), batch_key.lower()))
        print("===========there are {} {}s with this dataset=========".format(len(adata.obs[celltype_key].value_counts()), celltype_key.lower()))

    data_info = pd.crosstab(adata.obs[batch_key], adata.obs[celltype_key], margins=True, margins_name="Total")
    # display(data_info)
    return data_info

def prepare_anndata(adata, batch_key="batch", condition_key="condition"):
    
    """    
    ### Function: `prepare_anndata`

    Prepares and processes single-cell data in an `AnnData` object, calculating key metrics for batch correction and condition conservation. 

    #### Input:
    - `adata`: An AnnData object containing single-cell RNA-seq data with an expression matrix and metadata.
    - `batch_key`: Column name in `adata.obs` for batch labels (default is `"batch"`).
    - `condition_key`: Column name in `adata.obs` for condition labels (default is `"condition"`).

    #### Output:
    A dictionary (`input_dict`) containing the following keys:

    - **`exp`**: Expression matrix as a NumPy array.
    - **`meta`**: Metadata from `adata.obs`.
    - **`var`**: Variable (gene) information from `adata`.
    - **`batch_labels`**: Numeric codes for batch labels.
    - **`condition_labels`**: Numeric codes for condition labels.
    - **`num_features`**: Number of genes (columns in `exp`).
    - **`num_batches`**: Number of unique batches.
    - **`num_conditions`**: Number of unique conditions.
    - **`cond_ratio`**: Ratios for condition balance.
    - **`batch_weights`**: Weights for batch balance.
    - **`max_batch`**: Index of the batch with the largest number of cells.

    This output dictionary (`input_dict`) can be directly used as input to models like `scFLASH`.
    """
    # Prepare matrix data
    x = np.array(adata.X)
    print("Data matrix shape:", x.shape)
    
    # Set condition if None
    if condition_key is None:
        adata.obs[condition_key] = np.zeros(x.shape[0]).tolist()
    
    # Process metadata
    meta = adata.obs
    meta[condition_key] = meta[condition_key].astype('category')
    meta[batch_key] = meta[batch_key].astype('category')
    b = meta[batch_key].cat.codes.copy().values
    cond = meta[condition_key].cat.codes.copy().values
    
    # Calculate parameters
    num_features = x.shape[1]
    num_batches = len(np.unique(b))
    num_conditions = len(np.unique(cond))

    
    # Calculate condition weights and ratios
    cond_labels = cond
    num_cond_classes = len(np.unique(cond_labels))
    cond_probability_vector = np.bincount(cond_labels, minlength=num_cond_classes) / len(cond_labels)
    # cond_weights = 1 / cond_probability_vector
    # max_condition = np.argmax(cond_probability_vector)
    cond_ratio = np.array([cond_probability_vector[label] for label in cond_labels])

    print("Condition proportions:", cond_probability_vector)
    # print("Index of maximum condition count:", max_condition)
    print("Condition ratio vector:", cond_ratio)
    
    # Calculate batch weights and ratios
    batch_labels = b
    num_batch_classes = len(np.unique(batch_labels))
    batch_probability_vector = np.bincount(batch_labels, minlength=num_batch_classes) / len(batch_labels)
    batch_weights = 1 / batch_probability_vector
    max_batch = np.argmax(batch_probability_vector)

    print("Batch proportions:", batch_probability_vector)
    print("Index of maximum batch count:", max_batch)
    
    # Compile results in a dictionary
    input_dict = {
        "exp": x,
        "meta": meta,
        "var": adata.var,
        "batch_labels": b,
        "condition_labels": cond,
        "num_features": num_features,
        "num_batches": num_batches,
        "num_conditions": num_conditions,
        "cond_ratio": 1/cond_ratio,
        "batch_weights": batch_weights,
        "max_batch": max_batch
    }
    
    return input_dict

