import os
import sys
import time
import shutil
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
from my_parser import args
from model import *

sc.settings.set_figure_params(
    dpi=120, dpi_save=250, facecolor='white', fontsize=10, figsize=(10, 10))


def train(model: torch.nn.Module, adata: anndata.AnnData, args,
          device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')):

    # Samplers
    if args.n_samplers == 1:
        cell_gene_sampler = CellSampler(adata, args)
    else:
        cell_gene_sampler = CellSamplerPool(args.n_samplers, adata, args)
    samplers = [cell_gene_sampler]
    for sampler in samplers:
        sampler.start()

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
        
    # set up initial learning rate and optimizer
    lr = args.lr * (np.exp(-args.lr_decay) ** step)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # restore model
    if args.restore_step:
        model.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, f'model-{args.restore_step}')))
        optimizer.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, f'opt-{args.restore_step}')))
        print('Parameters restored.')
        train_instance_name = Path(args.ckpt_dir).name
        ckpt_dir = args.ckpt_dir
    elif args.restore_epoch:
        model.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, f'model-{args.restore_epoch}')))
        optimizer.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, f'opt-{args.restore_epoch}')))
        print('Parameters restored.')
        train_instance_name = Path(args.ckpt_dir).name
        ckpt_dir = args.ckpt_dir
    else:
        train_instance_name = get_train_instance_name(args)
        ckpt_dir = os.path.join(args.ckpt_dir, train_instance_name)
    next_ckpt_epoch = int(np.ceil(epoch / args.log_every) * args.log_every)


    # logging
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        for file in glob('*.py'):
            shutil.copy(file, os.path.join(ckpt_dir, file))
    logging([
        ('ds', args.dataset_str),
        ('bs', args.batch_size),
        ('lr', lr),
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
            new_tracked_items['max_norm'] = norms.max().cpu().numpy()
            optimizer.step()

            # log tracked items
            for key, val in new_tracked_items.items():
                tracked_items[key].append(val)

            step += 1
            if args.lr_decay:
                lr = lr * np.exp(-args.lr_decay)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            epoch = step / steps_per_epoch

            # eval
            if not args.no_eval and (epoch >= next_ckpt_epoch or step == args.updates or epoch >= args.n_epochs):
                print(time.strftime('%m_%d-%H_%M_%S') + ' ' * 20)

                model.eval()
                cell_types, metadata = model.get_cell_type(cell_gene_sampler, adata, args)

                # display log and save embedding visualization
                embeddings = model.get_cell_emb_weights()
                logging_items = get_logging_items(
                    embeddings, next_ckpt_epoch, lr, args, adata,
                    tracked_items, tracked_metric, cell_types, metadata)
                logging(logging_items, ckpt_dir)
                draw_embeddings(
                    adata=adata, epoch=next_ckpt_epoch, args=args, cell_types=cell_types,
                    embeddings=embeddings,
                    train_instance_name=train_instance_name, ckpt_dir=ckpt_dir)

                # checkpointing
                torch.save(model.state_dict(), os.path.join(
                    ckpt_dir, f'model-{next_ckpt_epoch}'))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_dir, f'opt-{next_ckpt_epoch}'))

                next_ckpt_epoch += args.log_every

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


def evaluate(model: torch.nn.Module, adata: anndata.AnnData, args):
    steps_per_epoch = max(adata.n_obs / args.batch_size, 1)
    if args.restore_step:
        step = args.restore_step
        epoch = step / steps_per_epoch
    elif args.restore_epoch:
        epoch = args.restore_epoch
        step = epoch * steps_per_epoch
    else:
        raise ValueError('Must specify step to restore')

    lr = args.lr * (np.exp(-args.lr_decay) ** step)

    if args.restore_step:
        model.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, f'model-{args.restore_step}')))
        print('Parameters restored.')
        train_instance_name = Path(args.ckpt_dir).name
        ckpt_dir = args.ckpt_dir
    elif args.restore_epoch:
        model.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, f'model-{args.restore_epoch}')))
        print('Parameters restored.')
        train_instance_name = Path(args.ckpt_dir).name
        ckpt_dir = args.ckpt_dir
    else:
        raise ValueError('Must specify checkpoint directory')

    model.eval()
    cell_types, metadata = model.get_cell_type(None, adata, args)

    # display log and save embedding visualization
    embeddings = model.get_cell_emb_weights()
    logging_items = get_logging_items(
        embeddings, int(epoch), lr, args, adata,
        defaultdict(list), defaultdict(dict), cell_types, metadata)
    save_embeddings(model, embeddings, args)
    draw_embeddings(
        adata=adata, epoch=int(epoch), args=args, cell_types=cell_types,
        embeddings=embeddings, train_instance_name=train_instance_name,
        ckpt_dir=ckpt_dir, fname_postfix="eval")
    logging(logging_items, ckpt_dir)


if __name__ == '__main__':
    matplotlib.use('Agg')
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
    elif args.dataframe_path:
        import pickle
        with open(args.dataframe_path, 'rb') as f:
            df = pickle.load(f)
            df.reset_index(drop=True, inplace=True)
        annotations = ['batch_id', 'cell_id', 'cell_type']
        if 'barcode' in df.columns:
            annotations.append('barcode')
        df_anno = df[annotations]
        df.drop(annotations, axis=1, inplace=True)
        adata = anndata.AnnData(X=df, obs=df_anno)
        args.dataset_str = Path(args.dataframe_path).stem
    else:
        adata = available_datasets[args.dataset_str].get_dataset(args)
    adata = process_dataset(adata, args)

    model_dict = dict(
        MixtureOfMultinomial=MixtureOfMultinomial,
        MixtureOfZINB=MixtureOfZINB,
        scETM=scETM,
        scETMMultiDecoder=scETMMultiDecoder
    )
    Model = model_dict[args.model]
    model = Model(adata, args)
    if torch.cuda.is_available():
        model = model.to(torch.device('cuda:0'))
    print([tuple(param.shape) for param in model.parameters()])
    if args.eval:
        evaluate(model, adata, args)
    else:
        train(model, adata, args)
