import pickle
import os
import scanpy as sc
import scvi.dataset
import anndata
import pandas as pd
import numpy as np


def process_dataset(adata, args):
    if args.norm_cell_read_counts:
        sc.pp.normalize_total(adata, target_sum=1e4)
    if args.quantile_norm:
        from sklearn.preprocessing import quantile_transform
        adata.X = quantile_transform(adata.X, axis=1, copy=True)
    if args.log1p:
        sc.pp.log1p(adata)

    col = list(map(lambda s: s.lower(), list(adata.obs.columns)))
    adata.obs.columns = col
    if 'cell_types' not in col and 'cell_type' in col:
        col[col.index('cell_type')] = 'cell_types'
    convert_batch_to_int = False
    if 'batch_indices' not in col and 'batch_id' in col:
        batches = list(adata.obs.batch_id.unique())
        batches.sort()
        if not isinstance(batches[-1], str) and batches[-1] + 1 == len(batches):
            col[col.index('batch_id')] = 'batch_indices'
        else:
            convert_batch_to_int = True
    adata.obs.columns = col
    if convert_batch_to_int:
        adata.obs['batch_indices'] = adata.obs.batch_id.apply(lambda x: batches.index(x))

    adata.obs_names_make_unique()

    if adata.obs.batch_indices.nunique() < 100:
        adata.obs.batch_indices = adata.obs.batch_indices.astype('str').astype('category')

    if args.pathway_csv_path:
        mat = pd.read_csv(args.pathway_csv_path, index_col=0)
        genes = sorted(list(set(mat.index).intersection(adata.var_names)))
        print(f'Found {len(genes)} mutual genes in the gene-pathway matrix and the dataset.')
        if args.gene_emb_dim == 0:
            # Fixed gene embedding only. Keep the genes in the intersection of "mat" and "adata".
            adata = adata[:, genes]
            adata.varm['gene_emb'] = mat.loc[genes, :].values
        else:
            # Will partly train gene embedding. Keep all genes from "adata".
            mat = mat.reindex(index = adata.var_names, fill_value=0.0)
            adata.varm['gene_emb'] = mat.values
    return adata


class DatasetConfig:
    def __init__(self, name, get_dataset):
        self.name = name
        self.get_dataset = lambda args: process_dataset(get_dataset(args), args)


cortex_config = DatasetConfig("cortex", lambda args: scvi.dataset.CortexDataset('../data/cortex').to_anndata())
prefrontal_cortex_config = DatasetConfig("prefrontalCortex", lambda args: scvi.dataset.PreFrontalCortexStarmapDataset('../data/PreFrontalCortex').to_anndata())

available_datasets = dict(cortex=cortex_config, prefrontalCortex=prefrontal_cortex_config)
