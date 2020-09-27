import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
sc.settings.verbosity = 3
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=120, dpi_save=250, facecolor='white', fontsize=10, figsize=(10, 10))

def umap_and_leiden(adata, save_path=False, use_rep=None,
                    leiden_resolution=0.35, visualize_batch=True, show=False):
    print(f'\n========== Resolution {leiden_resolution} ==========')
    color=['batch_indices', 'leiden', 'cell_types'] if visualize_batch else ['leiden', 'cell_types']
    if 'condition' in adata.obs:
        color.append('condition')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=leiden_resolution)
    print(f'Resolution: {leiden_resolution}, # clusters: {adata.obs.leiden.nunique()}')
    print(f'ARI_type: {adjusted_rand_score(adata.obs.cell_types, adata.obs.leiden)}')
    print(f'NMI_type: {normalized_mutual_info_score(adata.obs.cell_types, adata.obs.leiden)}')
    if visualize_batch:
        print(f'ARI_batch: {adjusted_rand_score(adata.obs.batch_indices, adata.obs.leiden)}')
        print(f'NMI_batch: {normalized_mutual_info_score(adata.obs.batch_indices, adata.obs.leiden)}')
    sc.pl.umap(adata, color=color, use_raw=False, save=save_path, show=show)
    plt.show()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--h5ad-path', type=str, help='path to h5ad file')
parser.add_argument('--adata-path', type=str, help='path to adata file')
parser.add_argument('--batch-removal', action='store_true', help="whether to do batch correction")
parser.add_argument('--ckpt-dir', type=str, help='path to checkpoint directory',
                    default=f"../results")
parser.add_argument('--model', type=str, choices=('VAE', 'LDVAE'), default='LDVAE',
                    help="model to use")
parser.add_argument('--restore', action='store_true', help='whether to restore from ckpt-dir')
parser.add_argument('--leiden-resolutions', type=float, nargs='*',
                    default=(0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2),
                    help='resolutions at leiden clustering')
args = parser.parse_args()

from pathlib import Path
if args.h5ad_path:
    adata = anndata.read_h5ad(args.h5ad_path)
    dataset_name = Path(args.h5ad_path).stem
elif args.adata_path:
    import pickle
    adata = pickle.load(args.adata_path)
    dataset_name = Path(args.adata_path).stem

col = list(map(lambda s: s.lower(), list(adata.obs.columns)))
adata.obs.columns = col
if 'cell_type' in col:
    col[col.index('cell_type')] = 'cell_types'
convert_batch_to_int = False
if 'batch_id' in col:
    batches = list(adata.obs.batch_id.unique())
    batches.sort()
    if not isinstance(batches[-1], str) and batches[-1] + 1 == len(batches):
        col[col.index('batch_id')] = 'batch_indices'
    else:
        convert_batch_to_int = True
adata.obs.columns = col
if convert_batch_to_int:
    adata.obs['batch_indices'] = adata.obs.batch_id.apply(lambda x: batches.index(x))

from scvi.dataset import AnnDatasetFromAnnData
dataset = AnnDatasetFromAnnData(adata)

from scvi.models import VAE, LDVAE
from scvi.inference import UnsupervisedTrainer, load_posterior
n_epochs = 400
lr = 1e-3
use_cuda = True

model_dict = dict(VAE=VAE, LDVAE=LDVAE)
if args.batch_removal:
    model = model_dict[args.model](dataset.nb_genes, n_batch=adata.obs.batch_indices.nunique())
else:
    model = model_dict[args.model](dataset.nb_genes)

if args.restore:
    full = load_posterior(args.ckpt_dir, model=model, use_cuda=use_cuda)
else:
    import os
    from time import strftime
    args.ckpt_dir = os.path.join(args.ckpt_dir, f"{dataset_name}_{args.model}_{strftime('%m_%d-%H_%M_%S')}")
    trainer = UnsupervisedTrainer(
        model,
        dataset,
        train_size=1.,
        use_cuda=use_cuda,
        frequency=5,
    )
    trainer.train(n_epochs=n_epochs, lr=lr)
    full = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)))
    full = full.update({"batch_size": 64})

    full.save_posterior(args.ckpt_dir)
    
latent, batch_indices, labels = full.sequential().get_latent()
batch_indices = batch_indices.ravel()

if 'batch_indices' in adata.obs:
    adata.obs.batch_indcies = adata.obs.batch_indices.astype(str).astype('category')

sc.settings.figdir = args.ckpt_dir
adata.obsm["X_scVI"] = latent
sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_scVI") 
for resolution in args.leiden_resolutions:
    umap_and_leiden(adata, save_path=f'_{dataset_name}_leiden{resolution}_{args.model}.pdf',
                    use_rep='X_scVI', leiden_resolution=resolution)

from sklearn.neighbors import NearestNeighbors
import scipy
def entropy_batch_mixing(latent_space, batches, n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    # code adapted from scGAN
    print('Calculating batch mixing entropy...', end='\r', flush=True)
    def entropy(hist_data):
        counts = pd.Series(hist_data).value_counts()
        freqs = counts / counts.sum()
        return (-freqs * np.log(freqs + 1e-20)).sum()

    n_neighbors = min(n_neighbors, len(latent_space) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(latent_space)
    kmatrix = nne.kneighbors_graph(
        latent_space) - scipy.sparse.identity(latent_space.shape[0])

    score = 0.
    for t in range(n_pools):
        indices = np.random.choice(
            np.arange(latent_space.shape[0]), size=n_samples_per_pool)
        score += np.mean(
            [
                entropy(
                    batches[
                        kmatrix[indices[i]].nonzero()[1]
                    ]
                )
                for i in range(n_samples_per_pool)
            ]
        )
    return score / n_pools
print(f'BE: {entropy_batch_mixing(latent, adata.obs.batch_indices)}')