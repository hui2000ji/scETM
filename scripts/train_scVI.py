import os
from time import strftime, time
import numpy as np
import scanpy as sc
import torch
import logging
import anndata
import psutil
import matplotlib
import argparse
from arg_parser import add_plotting_arguments
from scETM import initialize_logger, evaluate

logger = logging.getLogger(__name__)
initialize_logger(logger=logger)

if __name__ == '__main__':
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
    args = parser.parse_args()
        
    matplotlib.use('Agg')
    sc.settings.set_figure_params(
        dpi=args.dpi_show, dpi_save=args.dpi_save, facecolor='white', fontsize=args.fontsize, figsize=args.figsize)

    from pathlib import Path
    adata = anndata.read_h5ad(args.h5ad_path)
    dataset_name = Path(args.h5ad_path).stem

    if args.batch_removal:
        args.model = args.model + 'batch'
    if not args.restore:
        ckpt_dir = os.path.join(args.ckpt_dir, f"{dataset_name}_{args.model}_{strftime('%m_%d-%H_%M_%S')}")
        os.makedirs(ckpt_dir)


    from scvi.dataset import AnnDatasetFromAnnData
    from scvi.models import VAE, LDVAE
    from scvi.inference import UnsupervisedTrainer, load_posterior
    start_time = time()
    start_mem = psutil.Process().memory_info().rss
    logger.info(f'Before model instantiation and training: {psutil.Process().memory_info()}')

    dataset = AnnDatasetFromAnnData(adata)

    if args.model.startswith('VAE'):
        model = VAE(
            dataset.nb_genes,
            n_batch=adata.obs.batch_indices.nunique() if args.batch_removal else 0,
            n_latent=args.n_latent,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            dispersion="gene-batch" if args.batch_removal else "gene"
        )
    else:
        model = LDVAE(
            dataset.nb_genes,
            n_batch=adata.obs.batch_indices.nunique() if args.batch_removal else 0,
            n_latent=args.n_latent,
            n_hidden=args.n_hidden,
            n_layers_encoder=args.n_layers,
            dispersion="gene-batch" if args.batch_removal else "gene"
        )

    if args.restore:
        full = load_posterior(os.path.join(ckpt_dir, 'posterior'), model=model, use_cuda=torch.cuda.is_available())
    else:
        trainer = UnsupervisedTrainer(
            model,
            dataset,
            train_size=1.,
            use_cuda=torch.cuda.is_available(),
            frequency=5,
            batch_size=args.batch_size
        )
        trainer.train(n_epochs=args.n_epochs, lr=args.lr)
        full = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)))
        full = full.update({"batch_size": 64})

        full.save_posterior(os.path.join(ckpt_dir, 'posterior'))
    
    latent, batch_indices, labels = full.sequential().get_latent()
    batch_indices = batch_indices.ravel()

    time_cost = time() - start_time
    mem_cost = psutil.Process().memory_info().rss - start_mem
    logger.info(f'Duration: {time_cost:.1f} s ({time_cost / 60:.1f} min)')
    logger.info(f'After model instantiation and training: {psutil.Process().memory_info()}')

    if not args.no_eval:
        adata.obsm["scVI"] = latent
        result = evaluate(adata, embedding_key = "scVI", resolutions = args.resolutions, plot_dir = ckpt_dir)
        with open(os.path.join(args.ckpt_dir, 'table1.tsv'), 'a+') as f:
            # dataset, model, seed, ari, nmi, ebm, k_bet
            f.write(f'{args.dataset_str}\t{args.model}\t{args.seed}\t{result["ari"]}\t{result["nmi"]}\t{result["ebm"]}\t{result["k_bet"]}\t{time_cost}\t{mem_cost}\n', flush=True)
