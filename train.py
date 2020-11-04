import os
import sys
import time
import shutil
import psutil
import pickle
from pathlib import Path
from glob import glob
from collections import defaultdict

import matplotlib
import numpy as np
import anndata
import scanpy as sc
import torch
from torch import optim
from torch.utils.data import DataLoader

from edgesampler import CellSampler, CellSamplerPool
from train_utils import get_kl_weight, save_embeddings, \
    get_train_instance_name, logging, get_logging_items, draw_embeddings
from datasets import available_datasets, process_dataset
from my_parser import parser
from model import *


def train(model: torch.nn.Module, adata: anndata.AnnData, args, step=0, epoch=0,
          device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')):

    # Samplers
    if args.n_samplers == 1:
        cell_gene_sampler = CellSampler(adata, args)
    else:
        cell_gene_sampler = CellSamplerPool(args.n_samplers, adata, args)
    samplers = [cell_gene_sampler]
    for sampler in samplers:
        sampler.start()
        
    # set up initial learning rate and optimizer
    args.lr = args.lr * (np.exp(-args.lr_decay) ** step)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # restore model
    if args.restore_step:
        optimizer.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, f'opt-{args.restore_step}')))
        print('Optimizer restored.')
    elif args.restore_epoch:
        optimizer.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, f'opt-{args.restore_epoch}')))
        print('Optimizer restored.')

    ckpt_dir = args.ckpt_dir
    train_instance_name = Path(args.ckpt_dir).name
    next_ckpt_epoch = int(np.ceil(epoch / args.log_every) * args.log_every)


    # logging
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        for file in glob('*.py'):
            shutil.copy(file, os.path.join(ckpt_dir, file))
    logging([
        ('ds', args.dataset_str),
        ('bs', args.batch_size),
        ('lr', args.lr),
        ('lr-decay', args.lr_decay),
        ('n_cells', adata.n_obs),
        ('n_genes', adata.n_vars),
        ('n_edges', adata.X.sum()),
        ('n_labels', adata.obs.cell_types.nunique()),
        ('n_batches', adata.obs.batch_indices.nunique()),
        ('ckpt_dir', train_instance_name),
        ('true_label_dist', ', '.join(
            [f'{name}: {count}' for name, count in
             adata.obs.cell_types.value_counts().iteritems()])),
        ('argv', ' '.join(sys.argv))
    ], ckpt_dir)
    tracked_items = defaultdict(list)
    tracked_metric = defaultdict(dict)

    cell_types = None
    start_time = time.time()
    start_epoch = epoch
    try:
        while epoch < args.n_epochs:
            print(f'Training: Epoch {int(epoch):5d}/{args.n_epochs:5d}\tNext ckpt: {next_ckpt_epoch:7d}', end='\r')

            # construct hyper_param_dict
            hyper_param_dict = {
                'beta': get_kl_weight(args, epoch),
                'supervised_weight': args.max_supervised_weight
            }

            # construct data_dict
            data_dict = {k: v.to(device) for k, v in cell_gene_sampler.pipeline.get_message().items()}

            # train for one step
            model.train()
            optimizer.zero_grad()
            fwd_dict = model(data_dict, hyper_param_dict)
            loss, new_tracked_items = model.get_loss(
                fwd_dict, data_dict, hyper_param_dict)
            loss.backward()
            norms = torch.nn.utils.clip_grad_norm_(model.parameters(), 500)
            new_tracked_items['max_norm'] = norms.cpu().numpy()
            optimizer.step()

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
            if not args.no_eval and (epoch >= next_ckpt_epoch or step == args.updates or epoch >= args.n_epochs):
                duration = time.time() - start_time
                logging(f'Took {duration:.1f} seconds ({duration / 60:.1f} minutes) to train {epoch - start_epoch:.1f} epochs.', args.ckpt_dir)
                mem_info = psutil.Process().memory_info()
                logging(repr(mem_info), args.ckpt_dir)

                evaluate(model, adata, args, step, next_ckpt_epoch, args.save_embeddings and epoch >= args.n_epochs,
                         tracked_items, tracked_metric)

                # checkpointing
                torch.save(model.state_dict(), os.path.join(
                    ckpt_dir, f'model-{next_ckpt_epoch}'))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_dir, f'opt-{next_ckpt_epoch}'))

                next_ckpt_epoch += args.log_every
                start_time = time.time()
                start_epoch = epoch

        print("Optimization Finished: %s" % ckpt_dir)
    except:
        import traceback
        traceback.print_exc()
        print("Optimization Failed/Interrupted: %s" % ckpt_dir)
    finally:
        # join all threads
        for sampler in samplers:
            sampler.join(0.1)

        # keep only the best checkpoints, and delete the rest
        metric_list = sorted(list(
            tracked_metric[args.tracked_metric].items()), key=lambda x: x[1])
        if not metric_list:
            for metric in tracked_metric:
                if tracked_metric[metric]:
                    metric_list = sorted(list(
                        tracked_metric[metric].items()), key=lambda x: x[1])
                    break
        if len(metric_list) > 3:
            for epoch_, metric in metric_list[:-3]:
                if epoch_ == epoch:
                    continue
                os.remove(os.path.join(ckpt_dir, 'model-%d' % epoch_))
                os.remove(os.path.join(ckpt_dir, 'opt-%d' % epoch_))
        # save final inferred cell types
        if cell_types is not None:
            try:
                for cell_type_key in cell_types:
                    if cell_type_key.endswith('cell_type'):
                        prefix = cell_type_key[0]
                        cell_type = cell_types[cell_type_key]
                        with open(os.path.join(ckpt_dir, prefix +
                                            '_cell_type.pickle'), 'wb') as f_:
                            pickle.dump(cell_type, f_)
            except:
                import traceback
                traceback.print_exc()


def evaluate(model: torch.nn.Module, adata: anndata.AnnData, args, step, epoch,
             save_emb=False, tracked_items=defaultdict(list), tracked_metric=defaultdict(dict)):
    model.eval()
    cell_types, metadata = model.get_cell_type(None, adata, args)

    # display log and save embedding visualization
    embeddings = model.get_cell_emb_weights()
    logging_items = get_logging_items(
        embeddings, int(epoch), args, adata,
        tracked_items, tracked_metric, cell_types, metadata)
    if save_emb:
        save_embeddings(model, adata, embeddings, args)
    if not args.no_draw:
        draw_embeddings(
            adata=adata, epoch=int(epoch), args=args, cell_types=cell_types,
            embeddings=embeddings, ckpt_dir=args.ckpt_dir, fname_postfix="eval")
    logging(logging_items, args.ckpt_dir)


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

    if args.h5ad_path:
        adata = anndata.read_h5ad(args.h5ad_path)
        args.dataset_str = Path(args.h5ad_path).stem
    elif args.anndata_path:
        import pickle
        with open(args.anndata_path, 'rb') as f:
            adata = pickle.load(f)
        args.dataset_str = Path(args.anndata_path).stem
    else:
        adata = available_datasets[args.dataset_str].get_dataset(args)
    adata = process_dataset(adata, args)

    model_dict = dict(
        MixtureOfMultinomial=MixtureOfMultinomial,
        MixtureOfZINB=MixtureOfZINB,
        scETM=scETM,
        scETMMultiDecoder=scETMMultiDecoder,
        SupervisedClassifier=SupervisedClassifier
    )
    Model = model_dict[args.model]
    model = Model(adata, args)
    if torch.cuda.is_available():
        model = model.to(torch.device('cuda:0'))
    print([tuple(param.shape) for param in model.parameters()])

    # set up step and epoch
    steps_per_epoch = max(adata.n_obs / args.batch_size, 1)
    if args.restore_step:
        step = args.restore_step
        epoch = step / steps_per_epoch
    elif args.restore_epoch:
        epoch = args.restore_epoch
        step = epoch * steps_per_epoch
    else:
        step = epoch = 0

    if (args.updates and args.updates / steps_per_epoch <= args.kl_weight_anneal * 0.75) or args.n_epochs <= args.kl_weight_anneal * 0.75:
        args.kl_weight_anneal = epoch / 2

    # restore model
    if args.restore_step:
        model.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, f'model-{args.restore_step}')))
        print('Parameters restored.')
    elif args.restore_epoch:
        model.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, f'model-{args.restore_epoch}')))
        print('Parameters restored.')
    else:
        train_instance_name = get_train_instance_name(args)
        args.ckpt_dir = os.path.join(args.ckpt_dir, train_instance_name)

    if args.eval:
        evaluate(model, adata, args, step, epoch, args.save_embeddings)
    else:
        train(model, adata, args, step, epoch)