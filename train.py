from argparse import ArgumentError
import os
import sys
import time
from numpy.lib.arraysetops import isin
import psutil
import pickle
import logging
from pathlib import Path
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import torch
from torch import optim

from batch_sampler import CellSampler, CellSamplerPool
from train_utils import get_kl_weight, save_embeddings, clustering, \
    get_train_instance_name, draw_embeddings, entropy_batch_mixing
from datasets import available_datasets, process_dataset
from arg_parser import parser
from model import scETM


def train(model: torch.nn.Module, adata: anndata.AnnData, args, epoch=0,
          device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')):
    # sampler
    if args.n_samplers == 1 or args.batch_size >= adata.n_obs:
        sampler = CellSampler(adata, args)
    else:
        sampler = CellSamplerPool(adata, args)
    dataloader = iter(sampler)
        
    # set up initial learning rate and optimizer
    step = epoch * steps_per_epoch
    args.lr = args.lr * (np.exp(-args.lr_decay) ** step)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.restore_epoch:
        optimizer.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, f'opt-{args.restore_epoch}')))
        logging.debug('Optimizer restored.')

    tracked_items = defaultdict(list)

    next_ckpt_epoch = int(np.ceil(epoch / args.log_every) * args.log_every)

    while epoch < args.n_epochs:
        print(f'Training: Epoch {int(epoch):5d}/{args.n_epochs:5d}\tNext ckpt: {next_ckpt_epoch:7d}', end='\r')

        # construct hyper_param_dict
        hyper_param_dict = {
            'beta': get_kl_weight(args, epoch),
            'supervised_weight': args.max_supervised_weight
        }

        # construct data_dict
        data_dict = {k: v.to(device) for k, v in next(dataloader).items()}

        # train for one step
        new_tracked_items = model.train_step(optimizer, data_dict, hyper_param_dict)

        # log tracked items
        for key, val in new_tracked_items.items():
            tracked_items[key].append(val)

        step += 1
        if args.lr_decay:
            args.lr = args.lr * np.exp(-args.lr_decay)
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        epoch = step / steps_per_epoch

        # eval
        if epoch >= next_ckpt_epoch or epoch >= args.n_epochs:
            logging.info('=' * 10 + f'Epoch {epoch:.0f}' + '=' * 10)

            # log time and memory cost
            logging.info(repr(psutil.Process().memory_info()))
            if args.lr_decay:
                logging.info(f'lr: {args.lr}')

            # log statistics of tracked items
            for key, val in tracked_items.items():
                logging.info(f'{key}: {np.mean(val):7.4f}')
            tracked_items = defaultdict(list)
            
            if not args.no_eval:
                evaluate(model, adata, args, next_ckpt_epoch, args.save_embeddings and epoch >= args.n_epochs)

                # checkpointing
                torch.save(model.state_dict(), os.path.join(
                    args.ckpt_dir, f'model-{next_ckpt_epoch}'))
                torch.save(optimizer.state_dict(),
                        os.path.join(args.ckpt_dir, f'opt-{next_ckpt_epoch}'))

            logging.info('=' * 10 + f'End of evaluation' + '=' * 10)
            next_ckpt_epoch += args.log_every

    logging.info("Optimization Finished: %s" % args.ckpt_dir)
    if isinstance(sampler, CellSamplerPool):
        sampler.join(0.1)


def evaluate(model: scETM, adata: anndata.AnnData, args, epoch,
             save_emb=False):
    model.eval()

    embeddings = model.get_cell_emb_weights()
    for emb_name, emb in embeddings.items():
        adata.obsm[emb_name] = emb
    if 'cell_types' in adata.obs:
        cluster_key = clustering('delta', adata, args)
    else:
        cluster_key = None

    # Only calc BE at last step
    if adata.obs.batch_indices.nunique() > 1 and not args.no_be and \
            ((not args.eval and epoch == args.n_epochs) or (args.eval and epoch == args.restore_epoch)):
        for emb_name, latent_space in embeddings.items():
            if emb_name == 'recon_log':
                continue
            logging.info(f'{emb_name}_BE: {entropy_batch_mixing(latent_space, adata.obs.batch_indices):7.4f}')

    if not args.no_draw:
        color_by = args.color_by if cluster_key is None else ([cluster_key] + args.color_by)
        for emb_name, emb in embeddings.items():
            if emb_name == 'recon_log':
                continue
            draw_embeddings(adata=adata, fname=f'{args.dataset_str}_{args.model}_{emb_name}_epoch{epoch}.pdf',
                args=args, color_by=color_by, use_rep=emb_name)
    if save_emb:
        save_embeddings(model, adata, embeddings, args)


if __name__ == '__main__':
    args = parser.parse_args()
    matplotlib.use('Agg')
    sc.settings.set_figure_params(
        dpi=args.dpi_show, dpi_save=args.dpi_save, facecolor='white', fontsize=args.fontsize, figsize=args.figsize)
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    # load dataset
    if args.h5ad_path:
        adata = anndata.read_h5ad(args.h5ad_path)
        args.dataset_str = Path(args.h5ad_path).stem
    elif args.anndata_path:
        import pickle
        with open(args.anndata_path, 'rb') as f:
            adata = pickle.load(f)
        args.dataset_str = Path(args.anndata_path).stem
    elif args.dataset_str in available_datasets:
        adata = available_datasets[args.dataset_str].get_dataset(args)
    else:
        raise ValueError("Must specify dataset through one of h5ad_path, anndata_path, dataset_str.")

    # set up logger
    if not args.restore_epoch:
        train_instance_name = get_train_instance_name(args)
        args.ckpt_dir = os.path.join(args.ckpt_dir, train_instance_name)
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(args.ckpt_dir, 'log.txt'))
    logging.basicConfig(
        handlers=[stream_handler, file_handler],
        format='%(levelname)s [%(asctime)s]: %(message)s',
        level=logging.INFO
    )
    logging.info(f'argv: {" ".join(sys.argv)}')
    logging.info(f'ckpt_dir: {args.ckpt_dir}')
    logging.info(f'args: {repr(args)}')

    # process dataset
    adata = process_dataset(adata, args)
    logging.info(repr(psutil.Process().memory_info()))

    start_time = time.time()
    # build model
    model = scETM(adata, args)
    if torch.cuda.is_available():
        model = model.to(torch.device('cuda:0'))
    if args.restore_epoch:
        model.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, f'model-{args.restore_epoch}')))
        logging.debug('Parameters restored.')

    # set up step and epoch
    steps_per_epoch = max(adata.n_obs / args.batch_size, 1)
    epoch = args.restore_epoch if args.restore_epoch else 0
    if args.n_epochs <= args.n_warmup_epochs * 2:
        args.n_warmup_epochs = epoch / 2

    # train or evaluate
    if args.eval:
        evaluate(model, adata, args, epoch, args.save_embeddings)
    else:
        train(model, adata, args, epoch)
    duration = time.time() - start_time
    logging.info(f'Duration: {duration:.1f} s ({duration / 60:.1f} min)')
