import anndata
import random
import scanpy as sc
from pathlib import Path
import scanpy as sc
import numpy as np
from time import strftime, time
import psutil
import os
import logging
import matplotlib
import harmonypy as hm
import argparse
from scETM import evaluate, initialize_logger
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
    parser.add_argument('--dim-red', type=int, default=50, help='reduce the raw data into this many features before integrating')
    parser.add_argument('--seed', type=int, default=-1, help='set seed')
    add_plotting_arguments(parser)
    args = parser.parse_args()

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    matplotlib.use('Agg')
    sc.settings.set_figure_params(
        dpi=args.dpi_show, dpi_save=args.dpi_save, facecolor='white', fontsize=args.fontsize, figsize=args.figsize)

    # load dataset
    adata = anndata.read_h5ad(args.h5ad_path)
    dataset_name = Path(args.h5ad_path).stem
    adata.obs_names_make_unique()
    ckpt_dir = os.path.join(args.ckpt_dir, f'{dataset_name}_Harmony_seed{args.seed}_{strftime("%m_%d-%H_%M_%S")}')
    os.makedirs(ckpt_dir)

    start_time = time()
    start_mem = psutil.Process().memory_info().rss
    logger.info(f'Before model instantiation and training: {psutil.Process().memory_info()}')

    # preprocess
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if args.dim_red:
        sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=3000)
        sc.pp.pca(adata, n_comps=args.dim_red, use_highly_variable=True)
        data_mat = adata.obsm['X_pca']
    else:
        sc.pp.scale(adata)
        data_mat = np.array(adata.X)

    ho = hm.run_harmony(data_mat, meta_data=adata.obs, vars_use=['batch_indices'], max_iter_harmony=100)

    time_cost = time() - start_time
    mem_cost = psutil.Process().memory_info().rss - start_mem
    logger.info(f'Duration: {time_cost:.1f} s ({time_cost / 60:.1f} min)')
    logger.info(f'After model instantiation and training: {psutil.Process().memory_info()}')

    emb = anndata.AnnData(X = ho.result().T, obs = adata.obs)
    emb.write_h5ad(os.path.join(ckpt_dir, f"{dataset_name}_Harmony_seed{args.seed}.h5ad"))

    if not args.no_eval:
        result = evaluate(emb, embedding_key = "X", resolutions = args.resolutions, plot_dir = ckpt_dir, plot_fname=f"{dataset_name}_Harmony_seed{args.seed}_eval")
        with open(os.path.join(args.ckpt_dir, 'table1.tsv'), 'a+') as f:
            # dataset, model, seed, ari, nmi, ebm, k_bet
            f.write(f'{dataset_name}\tHarmony\t{args.seed}\t{result["ari"]}\t{result["nmi"]}\t{result["asw"]}\t{result["ebm"]}\t{result["k_bet"]}\t{time_cost}\t{mem_cost/1024}\n')

