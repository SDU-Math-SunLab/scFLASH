import numpy as np  
import pandas as pd
import scanpy as sc
import pandas as pd
import anndata as ad 
import scib
import os
from tqdm import tqdm
from sklearn.metrics import silhouette_samples, silhouette_score
from .knn_f1 import calculate_knn_performance

def ASW(exp_mat, labels_true):
    """
    Calculates the Average Silhouette Width (ASW) for a given data matrix and label set.
    
    Parameters:
        - exp_mat (array-like): Data matrix where each row is a sample and columns are features.
        - labels_true (array-like): True labels for each sample in the data matrix.

    Returns:
        - float: Normalized average silhouette width for the dataset.
    """
    # If the number of features is greater than 200, apply PCA for dimensionality reduction
    if exp_mat.shape[1] > 200:
        adata = ad.AnnData(exp_mat) 
        sc.tl.pca(adata, svd_solver='arpack') 
        X = adata.obsm['X_pca'][:, :20]  # Use first 20 principal components
    else:
        X = exp_mat
    labels = labels_true 
    
    # Compute silhouette values for each sample
    silhouette_values = silhouette_samples(X, labels)
    
    # Compute average silhouette width for the dataset
    average_silhouette_width = silhouette_score(X, labels)
    average_silhouette_width = (average_silhouette_width + 1) / 2

    return average_silhouette_width


def metric(adata_raw, adata_int, batch_key="batch", celltype_key="celltype", condition_key="condition",
              out_type="full", cal_iLISI=True, save_adata=False, folder_name="scFALSH_output"):
    """
    Computes various metrics for evaluating biological conservation, batch correction and condition conservation
    in integrated single-cell data.
    
    Parameters:
        - adata_raw (AnnData): Original AnnData object with raw data.
        - adata_int (AnnData): Integrated AnnData object after batch correction.
        - batch_key (str, optional): Key in `obs` for batch information. Defaults to "batch".
        - celltype_key (str, optional): Key in `obs` for cell type labels. Defaults to "celltype".
        - condition_key (str, optional): Key in `obs` for condition labels. Defaults to "condition".
        - out_type (str, optional): Type of analysis output, either 'full', 'embed', or 'knn'. Defaults to "full".
        - save_adata (bool, optional): If True, saves the modified AnnData object. Defaults to False.
        - folder_name (str, optional): Directory name for saving the AnnData object if save_adata is True.

    Returns:
        - tuple: (AnnData, DataFrame) Modified AnnData object with computed metrics and a DataFrame of extracted metrics.
    """
  
    # Ensure variable names are consistent between adata_raw and adata_int
    adata_int.var_names = adata_raw.var_names
    adata_int.var_names

    # Set categories for batch, celltype, and condition columns in both datasets
    adata_int.obs[batch_key] = adata_int.obs[batch_key].astype('category')
    adata_raw.obs[batch_key] = adata_raw.obs[batch_key].astype('category')
    adata_int.obs[celltype_key] = adata_int.obs[celltype_key].astype('category')
    adata_raw.obs[celltype_key] = adata_raw.obs[celltype_key].astype('category')
    adata_int.obs[condition_key] = adata_int.obs[condition_key].astype('category')
    adata_raw.obs[condition_key] = adata_raw.obs[condition_key].astype('category')

    # Full output
    if out_type == "full":
        # Perform PCA and calculate neighbors for batch correction
        sc.tl.pca(adata_int, svd_solver='arpack')
        sc.pp.neighbors(adata_int, n_neighbors=15, n_pcs=15)
        sc.tl.umap(adata_int)

        # Compute metrics for the full dataset with specified settings
        df_metric = scib.me.metrics(
            adata_raw, adata_int, batch_key=batch_key, label_key=celltype_key, type_="full", embed='X_pca', n_cores=4,
            cluster_key="louvain_cluster", ari_=True, nmi_=True, silhouette_=True, clisi_=False, hvg_score_=False,
            pcr_=False, isolated_labels_=False, isolated_labels_asw_=False, graph_conn_=False, ilisi_=cal_iLISI, kBET_=False, verbose=False
        )
        df_metric.columns = ["Value"]
        # print(df_metric)
        
        # Compute condition metrics with additional settings
        df_cond_metric = scib.me.metrics(
            adata_raw, adata_int, batch_key=batch_key, label_key=condition_key, type_="full", embed='X_pca', n_cores=4,
            ari_=False, isolated_labels_=False, isolated_labels_asw_=False, silhouette_=True, clisi_=False, ilisi_=False
        )
        df_cond_metric.columns = ["condition_metric value"]

    # Embedding output #################################################################################################
    if out_type == "embed":
        # Calculate neighbors and UMAP embedding for embedded representation
        sc.pp.neighbors(adata_int, use_rep="X_emb")
        sc.tl.umap(adata_int)

        # Compute metrics for embedded dataset
        df_metric = scib.me.metrics(
            adata_raw, adata_int, batch_key=batch_key, label_key=celltype_key, type_="embed", embed='X_emb', n_cores=4,
            cluster_key="louvain_cluster", ari_=True, nmi_=True, silhouette_=True, clisi_=True, hvg_score_=False,
            pcr_=False, isolated_labels_=True, isolated_labels_asw_=True, graph_conn_=True, ilisi_=cal_iLISI, kBET_=False
        )
        df_metric.columns = ["Value"]
        # print(df_metric)
        
        # Compute condition metrics
        df_cond_metric = scib.me.metrics(
            adata_raw, adata_int, batch_key=batch_key, label_key=condition_key, type_="embed", embed='X_emb', n_cores=4,
            ari_=False, isolated_labels_=False, isolated_labels_asw_=False, silhouette_=True, clisi_=False, ilisi_=False
        )
        df_cond_metric.columns = ["condition_metric value"]

    # KNN output ########################################################################################################
    if out_type == "knn":
        # UMAP embedding for KNN evaluation
        sc.tl.umap(adata_int)

        # Compute metrics using KNN
        df_metric = scib.me.metrics(
            adata_raw, adata_int, batch_key=batch_key, label_key=celltype_key, type_="full", embed='X_pca', n_cores=4,
            cluster_key="louvain_cluster", ari_=True, nmi_=True, silhouette_=False, clisi_=True, hvg_score_=False,
            pcr_=False, isolated_labels_=True, isolated_labels_asw_=False, graph_conn_=True, ilisi_=cal_iLISI, kBET_=False
        )
        df_metric.columns = ["Value"]
        # print(df_metric)
        
        # Compute condition metrics for KNN
        df_cond_metric = scib.me.metrics(
            adata_raw, adata_int, batch_key=batch_key, label_key=condition_key, type_="full", embed='X_pca', n_cores=4,
            ari_=False, isolated_labels_=True, isolated_labels_asw_=False, silhouette_=False, clisi_=False, ilisi_=False
        )
        df_cond_metric.columns = ["condition_metric value"]

    # Plot UMAP results for various groupings
    color_group = [celltype_key, batch_key, condition_key, "louvain_cluster"]
    sc.pl.umap(adata_int, color=color_group, legend_fontsize=10, wspace=0.3, frameon=False)

    # Separate AnnData object into subsets based on cell types
    adata1 = adata_int.copy()
    ct_categories = adata1.obs[celltype_key].cat.categories
    ct_names = ct_categories.tolist()
    adata_ls = []
    for ct_name in ct_names:
        adata_subset = adata1[adata1.obs[celltype_key] == ct_name].copy()
        adata_ls.append(adata_subset)

    # Calculate ASW for each subset of adata
    result_vector = []
    for i, adata_tep in enumerate(tqdm(adata_ls, desc="Calculate ASW_cond for each cell type")):
        if out_type == "embed":
            result = ASW(adata_tep.obsm["X_emb"], adata_tep.obs[condition_key].values)
        else:
            result = ASW(adata_tep.X, adata_tep.obs[condition_key].values)
        result_vector.append(result)

    # Average ASW values for condition analysis
    result_array = np.array(result_vector)
    average_result_array = sum(result_array) / len(result_array)

    # Calculate F1 score for KNN-based performance
    print("Calculating KNN-based performance...")
    f1_micro = calculate_knn_performance(adata_int, condition=condition_key)

    # Prepare metric labels and ASW values for DataFrame
    labels = ct_names + ["condition_metric value"]
    ASW_values = result_vector + [average_result_array]

    data = {'Labels': labels, 'ASW_condition': ASW_values}  
    df = pd.DataFrame(data).T
    df_transposed = df.iloc[1:4].rename(columns=df.iloc[0])

    # Append condition metric value
    # print(df_transposed)
    df_cond_metric = pd.concat([df_cond_metric, df_transposed[["condition_metric value"]]])
    df_cond_metric.loc["f1_score"] = [f1_micro]
    
    
    if cal_iLISI == True:
        # Extract specific metrics and add additional scores
        index = ['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label/batch', 'iLISI']
        df_extracted = df_metric.loc[index]
        df_extracted.loc["ASW_condition"] = df_transposed["condition_metric value"].values
        df_extracted.loc["f1_score"] = [f1_micro]

        # Update metric types for extracted metrics
        df_extracted['Metric Type'] = [
            'Bio conservation', 'Bio conservation', 'Batch correction', 'Batch correction', 
            'Cond conservation', 'Cond conservation'
        ]
        df_extracted = df_extracted[['Metric Type', "Value"]]
        
        df_extracted.index = ['NMI', 'ARI', 'ASW_batch', 'iLISI',"ASW_cond",'cond_knn'] 
        
    else:
        # Extract specific metrics and add additional scores
        index = ['NMI_cluster/label', 'ARI_cluster/label', 'ASW_label/batch']
        df_extracted = df_metric.loc[index]
        df_extracted.loc["ASW_condition"] = df_transposed["condition_metric value"].values
        df_extracted.loc["f1_score"] = [f1_micro]

        # Update metric types for extracted metrics
        df_extracted['Metric Type'] = [
            'Bio conservation', 'Bio conservation', 'Batch correction', 'Batch correction', 'Cond conservation'
        ]
        df_extracted = df_extracted[['Metric Type', "Value"]]
        
        df_extracted.index = ['NMI', 'ARI', 'ASW_batch', "ASW_cond",'cond_knn'] 
        
    # Convert DataFrame to string format
    df1 = df_extracted.applymap(str)

    # Save metrics in AnnData object
    adata_int.uns["Metric"] = df1

    # Save AnnData to file if requested
    if save_adata:
        os.makedirs(folder_name, exist_ok=True) 
        adata_int.write(f"{folder_name}.h5ad")

    return adata_int, df_extracted
