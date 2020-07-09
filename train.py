import os
import sys
import shutil
import pickle
from pathlib import Path
from glob import glob
from collections import defaultdict

import numpy as np
import anndata
import scanpy as sc
import torch
from torch import optim

from edgesampler import NonZeroEdgeSampler, EdgeSampler, VAEdgeSampler, \
    VAEdgeSamplerPool
from train_utils import get_beta, get_epsilon, get_eta, \
    get_train_instance_name, logging, get_logging_items, draw_embeddings
from datasets import available_datasets
from my_parser import parser
from model import CellGeneModel

sc.settings.set_figure_params(
    dpi=120, dpi_save=300, facecolor='white', fontsize=10, figsize=(10, 10))


def train(model, adata: anndata.AnnData, args,
          device=torch.device(
              "cuda:0" if torch.cuda.is_available() else "cpu")):
    # set up initial learning rate and optimizer
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    gumbel_tau = args.gumbel_max

    # Samplers
    if args.alias_sampling:
        # cell_gene_sampler = VAEdgeSampler(adata, args)
        cell_gene_sampler = VAEdgeSamplerPool(2, adata, args)
    else:
        cell_gene_sampler = NonZeroEdgeSampler(adata, args)
    samplers = [cell_gene_sampler]
    if args.max_delta:
        cc_edges = ((adata.X/adata.X.sum(1, keepdims=True))@
                    (adata.X/adata.X.sum(0, keepdims=True)).T).flatten()
        cell_sampler = EdgeSampler(
            cc_edges, args.batch_size, adata.n_obs, device)
        samplers.append(cell_sampler)
    if args.max_gamma:
        gg_edges = ((adata.X/adata.X.sum(0, keepdims=True)).T@
                    (adata.X/adata.X.sum(1, keepdims=True))).flatten()
        gene_sampler = EdgeSampler(
            gg_edges, args.batch_size, adata.n_vars, device)
        samplers.append(gene_sampler)
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
            print('Training: Step {:7d}/{:7d}\tNext ckpt: {:7d}'.format(
                step, args.updates, next_checkpoint), end='\r')

            # construct data_dict
            data_dict = cell_gene_sampler.pipeline.get_message()
            if args.max_gamma:
                g1, g2 = gene_sampler.pipeline.get_message()
                data_dict['g1'], data_dict['g2'] = g1, g2
            if args.max_delta:
                c1, c2 = cell_sampler.pipeline.get_message()
                data_dict['c1'], data_dict['c2'] = c1, c2
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
                'neg_weight': args.neg_weight
            }

            # train for one step
            model.train()
            optimizer.zero_grad()
            fwd_dict = model(data_dict, hyper_param_dict)
            loss, other_tracked_items = model.get_loss(
                fwd_dict, data_dict, hyper_param_dict)
            loss.backward()
            optimizer.step()
            step += 1
            gumbel_tau = np.maximum(
                gumbel_tau * np.exp(-args.gumbel_anneal),
                args.gumbel_min)

            # log tracked items
            tracked_items['loss'].append(loss.item())
            for key, val in other_tracked_items.items():
                tracked_items[key].append(val.item())

            # eval
            if step >= next_checkpoint or step == args.updates:
                model.eval()
                cell_types = model.get_cell_type(cell_gene_sampler)
                next_checkpoint += args.log_every

                # display log and save embedding visualization
                items = get_logging_items(
                    step, lr, hyper_param_dict['tau'], args, adata,
                    tracked_items, tracked_metric, cell_types)
                logging(items, ckpt_dir)
                draw_embeddings(
                    adata=adata, step=step, args=args, cell_types=cell_types,
                    embeddings=model.get_cell_emb_weights().items(),
                    train_instance_name=train_instance_name, ckpt_dir=ckpt_dir)

                # checkpointing
                torch.save(model.state_dict(), os.path.join(
                    ckpt_dir, 'model-%d' % step))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_dir, 'opt-%d' % step))

                # lr decay
                if args.lr_decay:
                    lr = lr * args.lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
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
    args = parser.parse_args()
    adata = available_datasets[args.dataset_str].get_dataset(args)
    if not args.n_labels:
        args.n_labels = adata.obs.cell_types.nunique()
    if not args.eval_batches:
        args.eval_batches = int(np.round(1000000 / args.batch_size))

    model = CellGeneModel(adata, args)
    train(model, adata, args)
