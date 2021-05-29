import scanorama
import random
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
    parser.add_argument('--seed', type=int, default=-1, help='set seed')
    add_plotting_arguments(parser)
    args = parser.parse_args()

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    matplotlib.use('Agg')
    sc.settings.set_figure_params(
        dpi=args.dpi_show, dpi_save=args.dpi_save, facecolor='white', fontsize=args.fontsize, figsize=args.figsize)

    adata = anndata.read_h5ad(args.h5ad_path)
    dataset_name = Path(args.h5ad_path).stem
    adata.obs_names_make_unique()
    ckpt_dir = os.path.join(args.ckpt_dir, f'{dataset_name}_Scanorama_seed{args.seed}_{strftime("%m_%d-%H_%M_%S")}')
    os.makedirs(ckpt_dir)

    adatas = []
    for batch in adata.obs.batch_indices.unique():
        part = adata[adata.obs.batch_indices == batch, :].copy()
        if isinstance(part.X, csr_matrix):
            part.X = np.array(part.X.todense())
        adatas.append(part)
    adata = anndata.concat(adatas)
    start_time = time()
    start_mem = psutil.Process().memory_info().rss
    logger.info(f'Before model instantiation and training: {psutil.Process().memory_info()}')
    
    # Integration and batch correction.
    integrated = scanorama.integrate_scanpy(adatas, dimred=args.dim_red)
    # returns a list of 3 np.ndarrays with 100 columns.

    time_cost = time() - start_time
    mem_cost = psutil.Process().memory_info().rss - start_mem
    logger.info(f'Duration: {time_cost:.1f} s ({time_cost / 60:.1f} min)')
    logger.info(f'After model instantiation and training: {psutil.Process().memory_info()}')
    
    emb = anndata.AnnData(X = np.concatenate(integrated), obs = adata.obs)
    emb.write_h5ad(os.path.join(ckpt_dir, f"{dataset_name}_Scanorama_seed{args.seed}.h5ad"))

    if not args.no_eval:
        result = evaluate(emb, embedding_key = "X", resolutions = args.resolutions, plot_dir = ckpt_dir, plot_fname=f"{dataset_name}_Scanorama_seed{args.seed}_eval")
        with open(os.path.join(args.ckpt_dir, 'table1.tsv'), 'a+') as f:
            # dataset, model, seed, ari, nmi, ebm, k_bet
            f.write(f'{dataset_name}\tScanorama\t{args.seed}\t{result["ari"]}\t{result["nmi"]}\t{result["asw"]}\t{result["ebm"]}\t{result["k_bet"]}\t{time_cost}\t{mem_cost/1024}\n')
