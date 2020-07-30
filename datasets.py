import pickle
import os
import scanpy as sc
import scvi.dataset
import anndata
import pandas as pd


def load_tabula_muris(args):
    mat_path = '../data/TM/FACS.csv'
    anno_path = '../data/TM/annotations_facs.csv'
    save_path = '../data/TM/FACS.pickle'
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            dataset = pickle.load(f)
            return dataset.to_anndata()
    df = pd.read_csv(mat_path, index_col=0)
    df = df.iloc[:, df.values.sum(0) > 0]
    cell_annotations = pd.read_csv(anno_path, index_col=2)
    cell_types = cell_annotations.cell_ontology_class.unique()
    cell_type_to_label = {cell_type: i for i, cell_type in enumerate(cell_types)}
    labels = cell_annotations.cell_ontology_class.map(lambda cell_type: cell_type_to_label[cell_type]).values.astype(
        'int32')
    cell_attributes = cell_annotations.to_dict('list')
    dataset = scvi.dataset.GeneExpressionDataset()
    unique_barcodes = cell_annotations['plate.barcode'].unique()
    barcode_to_id = {barcode: i for i, barcode in enumerate(unique_barcodes)}
    batch_id = cell_annotations['plate.barcode'].map(lambda barcode: barcode_to_id[barcode]).values.astype('int32')
    dataset.populate_from_data(df.values, None, batch_id, labels, df.columns,
                               cell_types, cell_attributes)
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset.to_anndata()


def process_dataset(get_dataset, args):
    if args.subsample_genes < 0:
        args.subsample_genes = available_datasets[args.dataset_str].n_genes
    adata = get_dataset(args)
    if args.norm_cell_read_counts:
        sc.pp.normalize_total(adata, target_sum=1e4)
    if args.subsample_genes < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.subsample_genes)
        adata = adata[:, adata.var.highly_variable]
    if args.quantile_norm:
        from sklearn.preprocessing import quantile_transform
        adata.X = quantile_transform(adata.X, axis=1, copy=True)
    if args.log1p:
        sc.pp.log1p(adata)
    if adata.obs.batch_indices.nunique() < 100:
        adata.obs.batch_indices = adata.obs.batch_indices.astype('str').astype('category')
    return adata


class DatasetConfig:
    def __init__(self, name, n_genes, n_labels, get_dataset):
        self.name = name
        self.n_genes = n_genes
        self.n_labels = n_labels
        self.get_dataset = lambda args: process_dataset(get_dataset, args)


def get_HCL_adult_thyroid(args):
    import pickle
    with open('../data/HCL/AdultThyroid.pickle', 'rb') as f:
        return pickle.load(f).to_anndata()


def get_TM_pancreas(args):
    import pickle
    with open('../data/TM/FACS_pancreas.pickle', 'rb') as f:
        return pickle.load(f).to_anndata()


def get_cortex(args):
    return scvi.dataset.CortexDataset('../data/cortex',
                                      total_genes=(args.subsample_genes if args.subsample_genes else None)).to_anndata()


def get_mouse_pancreas(args):
    import pickle
    with open('../data/MousePancreas/scvi_dataset.pickle', 'rb') as f:
        return pickle.load(f).to_anndata()


cortex_config = DatasetConfig("cortex", 558, 7, get_cortex)
prefrontal_cortex_config = DatasetConfig("prefrontalCortex", 158, 16, lambda args:
                                         scvi.dataset.PreFrontalCortexStarmapDataset(save_path='../data/PreFrontalCortex'))
tabula_muris_config = DatasetConfig(
    "TM", 23433, 82, lambda args: load_tabula_muris(args))
HCL_adult_thyroid_config = DatasetConfig(
    "HCLAdultThyroid", 24411, 8, get_HCL_adult_thyroid)
TM_pancreas_config = DatasetConfig("TMPancreas", 23043, 9, get_TM_pancreas)
mouse_pancreas_config = DatasetConfig(
    "MousePancreas", 14878, 13, get_mouse_pancreas)
available_datasets = dict(cortex=cortex_config, prefrontalCortex=prefrontal_cortex_config, TM=tabula_muris_config,
                          HCLAdultThyroid=HCL_adult_thyroid_config, TMPancreas=TM_pancreas_config, mousePancreas=mouse_pancreas_config)
