import anndata as ad
import scanpy as sc
from pathlib import Path
import scanpy as sc
from time import strftime, time
import psutil
import os
import logging
import argparse
from arg_parser import add_plotting_arguments, add_preprocessing_arguments
from datasets import process_dataset
from train_utils import draw_embeddings, clustering, entropy_batch_mixing

parser = argparse.ArgumentParser()
parser.add_argument('--h5ad-path', type=str, required=True, help='path to h5ad file')
parser.add_argument('--ckpt-dir', type=str, help='path to checkpoint directory',
                    default=os.path.join('..', 'results'))
parser.add_argument('--no-be', action='store_true', help='do not calculate batch mixing entropy')
parser.add_argument('--no-eval', action='store_true', help='quit immediately after training')
add_plotting_arguments(parser)
add_preprocessing_arguments(parser)
args = parser.parse_args()
sc.settings.set_figure_params(
    dpi=args.dpi_show, dpi_save=args.dpi_save, facecolor='white', fontsize=args.fontsize, figsize=args.figsize)

adata = ad.read_h5ad(args.h5ad_path)
dataset_name = Path(args.h5ad_path).stem

args.ckpt_dir = os.path.join(args.ckpt_dir, f'{dataset_name}_Harmony_{strftime("%m_%d-%H_%M_%S")}')
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

start_time = time()
logging.info(f'Before model instantiation and training: {psutil.Process().memory_info()}')

# preprocess
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)
sc.pp.pca(adata, n_comps=50, use_highly_variable=True)

metadata = adata.obs
import harmonypy as hm
vars_use = ['batch_indices']
from time import time

ho = hm.run_harmony(adata.obsm['X_pca'], metadata, vars_use, max_iter_harmony=100)

duration = time() - start_time
logging.info(f'Duration: {duration:.1f} s ({duration / 60:.1f} min)')


with open('/proc/self/status') as f:
    rss = f.read().split('VmRSS:')[1].split('\n')[0]
with open('/proc/self/status') as f:
    vmpeak = f.read().split('VmPeak:')[1].split('\n')[0]

logging.info('RSS: ' + rss.strip())
logging.info('peak: ' + vmpeak.strip())
logging.info(f'After model instantiation and training: {psutil.Process().memory_info()}')

if args.no_eval:
    import sys
    sys.exit(0)

# Evaluation
adata.obsm["Harmony"] = ho.result().T
cluster_key = clustering('Harmony', adata, args)
if adata.obs.batch_indices.nunique() > 1 and not args.no_be:
    logging.info(f'BE: {entropy_batch_mixing(adata.obsm["Harmony"], adata.obs.batch_indices):7.4f}')
if not args.no_draw:
    color_by = [cluster_key] + args.color_by
    draw_embeddings(adata=adata, fname=f'{dataset_name}_Harmony.pdf',
        args=args, color_by=color_by, use_rep='Harmony')

