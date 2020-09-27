import os
import sys
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

from edgesampler import NonZeroEdgeSampler, VAEdgeSampler, \
    VAEdgeSamplerPool, CellSampler, CellSamplerPool
from train_utils import get_beta, get_epsilon, get_eta, \
    get_train_instance_name, logging, get_logging_items, draw_embeddings
from datasets import available_datasets, process_dataset
from my_parser import args
from model import *

sc.settings.set_figure_params(
    dpi=120, dpi_save=250, facecolor='white', fontsize=10, figsize=(10, 10))


def train(model, adata: anndata.AnnData, args,
          device=torch.device(
              "cuda:0" if torch.cuda.is_available() else "cpu")):
    # set up initial learning rate and optimizer
    lr = args.lr * (np.exp(-args.lr_decay) ** (args.restore_step))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    gumbel_tau = max(args.gumbel_min, args.gumbel_max * (np.exp(-args.gumbel_anneal) ** args.restore_step))

    # Samplers
    if args.cell_sampling:
        if args.n_samplers == 1:
            cell_gene_sampler = CellSampler(adata, args)
        else:
            cell_gene_sampler = CellSamplerPool(args.n_samplers, adata, args)
    elif not args.no_alias_sampling:
        if args.n_samplers == 1:
            cell_gene_sampler = VAEdgeSampler(adata, args)
        elif args.n_samplers > 1:
            cell_gene_sampler = VAEdgeSamplerPool(args.n_samplers, adata, args)
    else:
        cell_gene_sampler = NonZeroEdgeSampler(adata, args)
    samplers = [cell_gene_sampler]
    # if args.max_delta:
    #     cc_edges = ((adata.X/adata.X.sum(1, keepdims=True))@
    #                 (adata.X/adata.X.sum(0, keepdims=True)).T).flatten()
    #     cell_sampler = EdgeSampler(
    #         cc_edges, args.batch_size, adata.n_obs, device)
    #     samplers.append(cell_sampler)
    # if args.max_gamma:
    #     gg_edges = ((adata.X/adata.X.sum(0, keepdims=True)).T@
    #                 (adata.X/adata.X.sum(1, keepdims=True))).flatten()
    #     gene_sampler = EdgeSampler(
    #         gg_edges, args.batch_size, adata.n_vars, device)
    #     samplers.append(gene_sampler)
    for sampler in samplers:
        sampler.start()

    # set up step, restore model, optimizer if required
    step = args.restore_step
    next_checkpoint = int(np.ceil(step / args.log_every) * args.log_every)
    if args.restore_step:
        model.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, 'model-%d' % args.restore_step)))
        optimizer.load_state_dict(torch.load(os.path.join(
            args.ckpt_dir, 'opt-%d' % args.restore_step)))
        print('Parameters restored.')
        train_instance_name = Path(args.ckpt_dir).name
        ckpt_dir = args.ckpt_dir
    else:
        train_instance_name = get_train_instance_name(args, adata)
        ckpt_dir = os.path.join(args.ckpt_dir, train_instance_name)

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
        while step < args.updates:
            print(f'Training: Epoch {epoch:5d}/{args.n_epochs:5d} Step {step:7d}/{args.updates:7d}\tNext ckpt: {next_checkpoint:7d}', end='\r')

            # construct hyper_param_dict
            hyper_param_dict = {
                'tau': gumbel_tau,
                'beta': get_beta(args, step),
                'gamma': args.max_gamma,
                'delta': args.max_delta,
                'epsilon': get_epsilon(args, step),
                'zeta': args.max_zeta,
                'eta': get_eta(args, step),
                'lambda': args.max_lambda,
                'neg_weight': args.neg_weight,
                'E': args.m_step and step % args.m_step == 0,
                'supervised_weight': args.max_supervised_weight
            }

            # construct data_dict
            if not args.m_step or hyper_param_dict['E']:
                data_dict = cell_gene_sampler.pipeline.get_message()
                # if args.max_gamma:
                #     g1, g2 = gene_sampler.pipeline.get_message()
                #     data_dict['g1'], data_dict['g2'] = g1, g2
                # if args.max_delta:
                #     c1, c2 = cell_sampler.pipeline.get_message()
                #     data_dict['c1'], data_dict['c2'] = c1, c2

            # train for one step
            model.train()
            optimizer.zero_grad()
            fwd_dict = model(data_dict, hyper_param_dict)
            loss, other_tracked_items = model.get_loss(
                fwd_dict, data_dict, hyper_param_dict)
            loss.backward()
            norms = torch.nn.utils.clip_grad_norm_(model.parameters(), 500)
            other_tracked_items['max_norm'] = norms.max()
            optimizer.step()

            # log tracked items
            tracked_items['loss'].append(loss.detach().item())
            for key, val in other_tracked_items.items():
                tracked_items[key].append(val.detach().item())

            step += 1
            gumbel_tau = np.maximum(
                gumbel_tau * np.exp(-args.gumbel_anneal),
                args.gumbel_min)
            if args.lr_decay:
                lr = lr * np.exp(-args.lr_decay)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # eval
            if step >= next_checkpoint or step == args.updates:
                model.eval()
                cell_types = model.get_cell_type(cell_gene_sampler, adata, args)
                next_checkpoint += args.log_every

                # display log and save embedding visualization
                embeddings = model.get_cell_emb_weights()
                items = get_logging_items(
                    embeddings, step, lr, hyper_param_dict['tau'], args, adata,
                    tracked_items, tracked_metric, cell_types)
                logging(items, ckpt_dir)
                draw_embeddings(
                    adata=adata, step=step, args=args, cell_types=cell_types,
                    embeddings=embeddings,
                    train_instance_name=train_instance_name, ckpt_dir=ckpt_dir)

                # checkpointing
                torch.save(model.state_dict(), os.path.join(
                    ckpt_dir, 'model-%d' % step))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_dir, 'opt-%d' % step))

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
            for step_, metric in metric_list[:-3]:
                if step_ == step:
                    continue
                os.remove(os.path.join(ckpt_dir, 'model-%d' % step_))
                os.remove(os.path.join(ckpt_dir, 'opt-%d' % step_))
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

    args.updates = args.n_epochs * adata.n_obs / args.batch_size

    model_dict = dict(
        vGraph=CellGeneModel,
        vGraphEM=vGraphEM,
        LINE=LINE,
        MixtureOfMultinomial=MixtureOfMultinomial,
        MixtureOfZINB=MixtureOfZINB,
        scETM=scETM,
        NewModel=NewModel,
        scETMMultiDecoder=scETMMultiDecoder
    )
    Model = model_dict[args.model]
    model = Model(adata, args).to(torch.device('cuda:0'))
    print([tuple(param.shape) for param in model.parameters()])
    train(model, adata, args)
