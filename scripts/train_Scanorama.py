import scanorama
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
from time import strftime, time
import psutil
import anndata
import os
import logging
import matplotlib
from scETM import evaluate, initialize_logger
import argparse
from arg_parser import add_plotting_arguments

logger = logging.getLogger(__name__)
initialize_logger(logger=logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5ad-path', type=str, required=True, help='path to h5ad file')
    parser.add_argument('--ckpt-dir', type=str, help='path to checkpoint directory',
                        default=os.path.join('..', 'results'))
    parser.add_argument('--no-be', action='store_true', help='do not calculate batch mixing entropy')
    parser.add_argument('--no-eval', action='store_true', help='quit immediately after training')
    parser.add_argument('--dim-red', type=int, default=100, help='reduce the raw data into this many features before integrating')
    add_plotting_arguments(parser)
    args = parser.parse_args()

    matplotlib.use('Agg')
    sc.settings.set_figure_params(
        dpi=args.dpi_show, dpi_save=args.dpi_save, facecolor='white', fontsize=args.fontsize, figsize=args.figsize)

    adata = anndata.read_h5ad(args.h5ad_path)
    dataset_name = Path(args.h5ad_path).stem
    adata.obs_names_make_unique()
    args.ckpt_dir = os.path.join(args.ckpt_dir, f'{dataset_name}_Scanorama_{strftime("%m_%d-%H_%M_%S")}')
    os.makedirs(args.ckpt_dir)

    start_time = time()
    adatas = []
    for batch in adata.obs.batch_indices.unique():
        part = adata[adata.obs.batch_indices == batch, :].copy()
        if isinstance(part.X, csr_matrix):
            part.X = np.array(part.X.todense())
        adatas.append(part)
    adata = anndata.concat(adatas)
    logger.info(f'Before model instantiation and training: {psutil.Process().memory_info()}')
    
    # Integration and batch correction.
    integrated = scanorama.integrate_scanpy(adatas, dimred=args.dim_red)
    # returns a list of 3 np.ndarrays with 100 columns.
    duration = time() - start_time
    logger.info(f'Duration: {duration:.1f} s ({duration / 60:.1f} min)')

    if os.path.exists('/proc/self/status'):
        with open('/proc/self/status') as f:
            text = f.read()
        rss = text.split('VmRSS:')[1].split('\n')[0]
        vmpeak = text.split('VmPeak:')[1].split('\n')[0]
        logger.info('RSS: ' + rss.strip())
        logger.info('peak: ' + vmpeak.strip())
    logger.info(f'After model instantiation and training: {psutil.Process().memory_info()}')

if not args.no_eval:
    adata.obsm["Scanorama"] = np.concatenate(integrated)
    evaluate(adata, embedding_key = "Scanorama", resolutions = args.resolutions, plot_dir = args.ckpt_dir)
