import os
from time import strftime
import numpy as np
import scanpy as sc
import logging
import anndata
import psutil
import matplotlib
matplotlib.use('Agg')

import argparse
from arg_parser import add_preprocessing_arguments, add_plotting_arguments
parser = argparse.ArgumentParser()
parser.add_argument('--h5ad-path', type=str, required=True, help='path to h5ad file')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of the model')
parser.add_argument('--batch-removal', action='store_true', help="whether to do batch correction")
parser.add_argument('--ckpt-dir', type=str, help='path to checkpoint directory',
                    default=os.path.join('..', 'results'))
parser.add_argument('--model', type=str, choices=('VAE', 'LDVAE'), default='LDVAE',
                    help="model to use")
parser.add_argument('--restore', action='store_true', help='whether to restore from ckpt-dir')
parser.add_argument('--no-be', action='store_true', help='do not calculate batch mixing entropy')
parser.add_argument('--no-eval', action='store_true', help='quit immediately after training')
parser.add_argument('--n-epochs', type=int, default=400, help="number of epochs to train")
parser.add_argument('--n-layers', type=int, default=1, help='number of encoder and decoder (if any) layers')
parser.add_argument('--n-hidden', type=int, default=128, help='hidden layer size')
parser.add_argument('--n-latent', type=int, default=10, help='latent variable size')
parser.add_argument('--batch-size', type=int, default=2000, help='Batch size for training')
add_plotting_arguments(parser)
add_preprocessing_arguments(parser)

args = parser.parse_args()
sc.settings.set_figure_params(
    dpi=args.dpi_show, dpi_save=args.dpi_save, facecolor='white', fontsize=args.fontsize, figsize=args.figsize)

from pathlib import Path
from datasets import process_dataset
adata = anndata.read_h5ad(args.h5ad_path)
dataset_name = Path(args.h5ad_path).stem

if args.batch_removal:
    args.model = args.model + 'batch'
if not args.restore:
    args.ckpt_dir = os.path.join(args.ckpt_dir, f"{dataset_name}_{args.model}_{strftime('%m_%d-%H_%M_%S')}")
    os.makedirs(args.ckpt_dir)
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(args.ckpt_dir, 'log.txt'))
for handler in (stream_handler, file_handler):
    handler.setFormatter(logging.Formatter('%(levelname)s [%(asctime)s]: %(message)s'))
    handler.setLevel(logging.INFO)
logger = logging.getLogger()
for handler in logger.handlers:
    logger.removeHandler(handler)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

adata = process_dataset(adata, args)
from scvi.dataset import AnnDatasetFromAnnData
from scvi.models import VAE, LDVAE
from scvi.inference import UnsupervisedTrainer, load_posterior
from time import time
start_time = time()

logging.info(f'Before model instantiation and training: {psutil.Process().memory_info()}')
dataset = AnnDatasetFromAnnData(adata)

use_cuda = True

model_dict = dict(VAE=VAE, LDVAE=LDVAE)
if args.model.startswith('VAE'):
    model = VAE(
        dataset.nb_genes,
        n_batch=adata.obs.batch_indices.nunique() if args.batch_removal else 0,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers
    )
else:
    model = LDVAE(
        dataset.nb_genes,
        n_batch=adata.obs.batch_indices.nunique() if args.batch_removal else 0,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_layers_encoder=args.n_layers
    )

if args.restore:
    full = load_posterior(os.path.join(args.ckpt_dir, 'posterior'), model=model, use_cuda=use_cuda)
else:
    trainer = UnsupervisedTrainer(
        model,
        dataset,
        train_size=1.,
        use_cuda=use_cuda,
        frequency=5,
        batch_size=args.batch_size
    )
    trainer.train(n_epochs=args.n_epochs, lr=args.lr)
    full = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)))
    full = full.update({"batch_size": 64})

    full.save_posterior(os.path.join(args.ckpt_dir, 'posterior'))
    
latent, batch_indices, labels = full.sequential().get_latent()
batch_indices = batch_indices.ravel()

duration = time() - start_time
logging.info(f'Duration: {duration:.1f} s ({duration / 60:.1f} min)')
logging.info(f'After model instantiation and training: {psutil.Process().memory_info()}')

if args.no_eval:
    import sys
    sys.exit(0)

# Evaluation
from train_utils import clustering, entropy_batch_mixing, draw_embeddings

adata.obsm["X_scVI"] = latent
cluster_key = clustering('X_scVI', adata, args)
if adata.obs.batch_indices.nunique() > 1 and not args.no_be:
    logging.info(f'BE: {entropy_batch_mixing(latent, adata.obs.batch_indices):7.4f}')
if not args.no_draw:
    color_by = [cluster_key] + args.color_by
    draw_embeddings(adata=adata, fname=f'{dataset_name}_{args.model}.pdf',
        args=args, color_by=color_by, use_rep='X_scVI')
