import scanpy as sc
import anndata

def process_dataset(get_dataset, args):
    if args.subsample_genes < 0:
        args.subsample_genes = available_datasets[args.dataset_str].n_genes
    adata = get_dataset(args)
    if args.subsample_genes < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.subsample_genes)
    if args.norm_cell_read_counts:
        sc.pp.normalize_total(adata, target_sum=1e4)
    if args.quantile_norm:
        from sklearn.preprocessing import quantile_transform
        adata.X = quantile_transform(adata.X, axis=1, copy=True)
    if args.log1p:
        sc.pp.log1p(adata)
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
        return pickle.load(f)

def get_TM_pancreas(args):
    import pickle
    with open('../data/TM/FACS_pancreas.pickle', 'rb') as f:
        return pickle.load(f)

def get_cortex(args):
    adata = scvi.dataset.CortexDataset('../data/cortex',
                                       total_genes=(args.subsample_genes if args.subsample_genes else None)).to_anndata()



cortex_config = DatasetConfig("cortex", 558, 7, get_cortex)
prefrontal_cortex_config = DatasetConfig("prefrontalCortex", 158, 16, lambda args:
    scvi.dataset.PreFrontalCortexStarmapDataset(save_path='../data/PreFrontalCortex'))
tabula_muris_config = DatasetConfig("TM", 23433, 82, lambda args: load_tabula_muris())
HCL_adult_thyroid_config = DatasetConfig("HCLAdultThyroid", 24411, 8, get_HCL_adult_thyroid)
TM_pancreas_config = DatasetConfig("TMPancreas", 23433, 9, get_TM_pancreas)
available_datasets = dict(cortex=cortex_config, prefrontalCortex=prefrontal_cortex_config, TM=tabula_muris_config,
                          HCLAdultThyroid=HCL_adult_thyroid_config, TMPancreas=TM_pancreas_config)