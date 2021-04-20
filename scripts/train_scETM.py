import psutil
from arg_parser import parser
import os
from time import time
import scanpy as sc
import scvi.dataset
import anndata
import pandas as pd
import logging
from pathlib import Path
from scETM import scETM, UnsupervisedTrainer, initialize_logger, evaluate
import matplotlib

logger = logging.getLogger(__name__)
initialize_logger(logger=logger)


if __name__ == '__main__':
    args = parser.parse_args()

    matplotlib.use('Agg')
    sc.settings.set_figure_params(
        dpi=args.dpi_show, dpi_save=args.dpi_save, facecolor='white', fontsize=args.fontsize, figsize=args.figsize)

    # load dataset
    if args.h5ad_path:
        adata = anndata.read_h5ad(args.h5ad_path)
        args.dataset_str = Path(args.h5ad_path).stem
    elif args.dataset_str == 'cortex':
        adata = scvi.dataset.CortexDataset('../data/cortex').to_anndata()
    elif args.dataset_str == 'hemato':
        adata = scvi.dataset.CortexDataset('../data/hemato').to_anndata()
    elif args.dataset_str == 'prefrontalCortex':
        adata = scvi.dataset.PreFrontalCortexStarmapDataset('../data/PreFrontalCortex').to_anndata()
    else:
        raise ValueError("Must specify dataset through one of h5ad_path, anndata_path, dataset_str.")

    adata.obs_names_make_unique()

    if hasattr(args, 'pathway_csv_path') and args.pathway_csv_path:
        mat = pd.read_csv(args.pathway_csv_path, index_col=0)
        if args.trainable_gene_emb_dim == 0:
            genes = sorted(list(set(mat.index).intersection(adata.var_names)))
            logger.info(f'Using {mat.shape[1]}-dim fixed gene embedding for {len(genes)} genes appeared in both the gene-pathway matrix and the dataset.')
            adata = adata[:, genes]
            adata.varm['gene_emb'] = mat.loc[genes, :].values
        else:
            logger.info(f'{mat.shape[1]} dimensions of the gene embeddings will be trainable. Keeping all genes in the dataset') 
            mat = mat.reindex(index = adata.var_names, fill_value=0.0)
            adata.varm['gene_emb'] = mat.values

    if hasattr(args, 'color_by') and (hasattr(args, 'no_draw') and not args.no_draw) and (hasattr(args, 'no_eval') and not args.no_eval):
        for col_name in args.color_by:
            assert col_name in adata.obs, f"{col_name} in args.color_by but not in adata.obs"

    start_time = time()
    start_mem = psutil.Process().memory_info().rss
    logger.info(f'Before model instantiation and training: {psutil.Process().memory_info()}')

    model = scETM(
        n_trainable_genes = adata.n_vars,
        n_batches = adata.obs.cell_types.nunique(),
        n_topics = args.n_topics,
        trainable_gene_emb_dim = args.trainable_gene_emb_dim,
        hidden_sizes = args.hidden_sizes,
        bn = not args.no_bn,
        dropout_prob = args.dropout_prob,
        normed_loss = args.normed_loss,
        enable_batch_bias = args.batch_bias
    )

    trainer = UnsupervisedTrainer(
        model,
        adata,
        train_instance_name = f"{args.dataset_str}_scETM{args.log_str}_seed{args.seed}",
        seed = args.seed,
        ckpt_dir = args.ckpt_dir,
        batch_size = args.batch_size,
        test_ratio = args.test_ratio,
        data_split_seed = args.data_split_seed,
        restore_epoch = args.restore_epoch,
        init_lr = args.lr,
        lr_decay = args.lr_decay
    )

    trainer.train(
        n_epochs = args.n_epochs,
        eval_every = args.eval_every,
        n_samplers = args.n_samplers,
        kl_warmup_ratio = args.kl_warmup_ratio,
        min_kl_weight = args.min_kl_weight,
        max_kl_weight = args.max_kl_weight,
        save_model_ckpt = not args.no_model_ckpt,
        eval = not args.no_eval,
        batch_col = 'cell_types',
        record_log_path = os.path.join(trainer.ckpt_dir, 'record.tsv'),
        eval_result_log_path = os.path.join(args.ckpt_dir, 'result.tsv'),
        eval_kwargs = dict(resolutions=args.resolutions, batch_col='batch_indices')
    )

    time_cost = time() - start_time
    mem_cost = psutil.Process().memory_info().rss - start_mem
    logger.info(f'Duration: {time_cost:.1f} s ({time_cost / 60:.1f} min)')
    logger.info(f'After model instantiation and training: {psutil.Process().memory_info()}')

    result = evaluate(model, adata,
        resolutions = args.resolutions,
        plot_fname = f'{trainer.train_instance_name}_{trainer.model.clustering_input}_eval',
        plot_dir = trainer.ckpt_dir
    )
    with open(os.path.join(args.ckpt_dir, 'table1.tsv'), 'a+') as f:
        # dataset, model, seed, ari, nmi, ebm, k_bet, time_cost, mem_cost
        f.write(f'{args.dataset_str}\tscETM\t{args.seed}\t{result["ari"]}\t{result["nmi"]}\t{result["ebm"]}\t{result["k_bet"]}\t{time_cost}\t{mem_cost}\n', flush=True)
