import time
import os

import matplotlib.pyplot as plt 
import anndata
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, \
    normalized_mutual_info_score


def get_train_instance_name(args, adata: anndata.AnnData):
    strs = [
        args.dataset_str,
        args.model + args.emb_combine
        if args.model.startswith('vGraph') else args.model]
    for tuple_ in (
            ('genes', args.subsample_genes, adata.n_vars),
            ('nLabels', args.n_labels, adata.obs.cell_types.nunique()),
            ('clip', args.clip),
            ('scale', args.scale),
            ('beta', args.max_beta, 1.),
            ('minBeta', args.min_beta),
            ('gamma', args.max_gamma),
            ('delta', args.max_delta),
            ('epsilon', args.max_epsilon),
            ('zeta', args.max_zeta),
            ('eta', args.max_eta),
            ('cycBeta', args.cyclic_anneal),
            ('linBeta', args.linear_anneal, 1200),
            ('linEpsilon', args.linear_anneal_epsilon),
            ('linEta', args.linear_anneal_eta),
            ('cLoss', args.g2c_factor, 1.),
            ('negSmpls', args.neg_samples, 5),
            ('negWeight', args.neg_weight, 1.)
    ):
        if len(tuple_) == 2:
            name, numeric_ = tuple_
            default = 0
        else:
            name, numeric_, default = tuple_
        if numeric_ != default:
            strs.append('%s%g' % (name, numeric_))
    for name, bool_ in (
            ('gumbel', args.gumbel),
            ('qn', args.quantile_norm),
            ('log1p', args.log1p),
            ('normRdCnt', args.norm_cell_read_counts),
            ('decpPQ', args.decouple_pq),
            ('aliasSpl', args.alias_sampling),
            (args.log_str, args.log_str)
    ):
        if bool_:
            strs.append(name)
    current_time = time.strftime('%m_%d-%H_%M_%S')
    strs.append('time%s' % current_time)
    train_instance_name = '_'.join(strs)
    return train_instance_name


def logging(logging_items, ckpt_dir, time_str=None):
    str = '; '.join(['{} {}'.format(key, val) for key, val in logging_items])
    if time_str is None:
        str = str + ' ' + time.strftime('%m-%d %H:%M:%S')
    else:
        str = str + ' ' + time_str
    with open(os.path.join(ckpt_dir, 'log.txt'), 'a+') as f:
        f.write(str + '\n')
    print(str)
    return str


def get_beta(args, step):
    if args.cyclic_anneal:
        cycle_len = args.cyclic_anneal
        idx_in_cycle = step % cycle_len
        beta = max(min(1., idx_in_cycle / (cycle_len * 0.6))
                   * args.max_beta, args.min_beta)
    elif args.linear_anneal:
        beta = max(min(1., step / args.linear_anneal)
                   * args.max_beta, args.min_beta)
    else:
        beta = args.max_beta
    return beta


def get_eta(args, step):
    if args.linear_anneal_eta:
        eta = min(1., step / args.linear_anneal_eta) * args.max_eta
    else:
        eta = args.max_eta
    return eta


def get_epsilon(args, step):
    if args.linear_anneal_epsilon:
        epsilon = min(1., step / args.linear_anneal_epsilon) * args.max_epsilon
    else:
        epsilon = args.max_epsilon
    return epsilon


def get_logging_items(step, lr, gumbel_tau, args, adata,
                      tracked_items, tracked_metric, cell_types):
    items = [('step', '%7d' % step), ('gumbel', '%6.4f' % gumbel_tau)]
    if args.lr_decay < 1.:
        items.append(('lr', '%7.2e' % lr))
    for key, val in tracked_items.items():
        items.append((key, '%7.4f' % np.mean(val)))
    for cell_type_key in cell_types:
        if cell_type_key.endswith('cell_type'):
            prefix = cell_type_key[0]
            cell_type = cell_types[cell_type_key]
            tracked_metric['%c_lbl' % prefix][step] = \
                [(cell_type == label).sum() for label in np.unique(cell_type)]
            tracked_metric['%c_nmi' % prefix][step] = \
                normalized_mutual_info_score(cell_type, adata.obs.cell_types)
            tracked_metric['%c_ari' % prefix][step] = \
                adjusted_rand_score(cell_type, adata.obs.cell_types)
            tracked_metric['%c_ami' % prefix][step] = \
                adjusted_mutual_info_score(cell_type, adata.obs.cell_types)
            items.extend([
                ('%c_lbl' % prefix, str(
                    tracked_metric['%c_lbl' % prefix][step])),
                ('%c_nmi' % prefix, '%7.4f' %
                 tracked_metric['%c_nmi' % prefix][step]),
                ('%c_ari' % prefix, '%7.4f' %
                 tracked_metric['%c_ari' % prefix][step]),
                ('%c_ami' % prefix, '%7.4f' %
                 tracked_metric['%c_ami' % prefix][step])])
        elif cell_type_key.endswith('gene_type'):
            prefix = cell_type_key[0]
            gene_type = cell_types[cell_type_key]
            items.append(('%cG_lbl' % prefix, str(
                [(gene_type == i).sum() for i in range(args.n_labels)])))
        else:
            raise ValueError('Invalid cell type key ' + cell_type_key)
    for key in tracked_items:
        tracked_items[key] = list()
    return items


def draw_embeddings(adata: anndata.AnnData, step: int, args, cell_types: dict,
                    embeddings: dict, train_instance_name: str, ckpt_dir: str,
                    save: bool = True, show: bool = False):
    cell_type_keys = []
    for cell_type_key in cell_types:
        if cell_type_key.endswith('cell_type'):
            prefix = cell_type_key.split('_')[0]
            cell_type = cell_types[cell_type_key]
            if prefix in args.always_draw or step == args.updates:
                cell_type_keys. append(prefix)
                adata.obs[prefix] = cell_type
                adata.obs[prefix] = adata.obs[prefix].astype(
                    'str').astype('category')
    cell_type_keys.append('cell_types')
    if adata.obs.batch_indices.nunique() > 1:
        cell_type_keys = ['batch_indices'] + cell_type_keys
    for emb_name, emb in embeddings:
        adata.obsm[emb_name] = emb
        sc.pp.neighbors(adata, use_rep=emb_name, n_neighbors=args.n_neighbors)
        sc.tl.umap(adata, min_dist=args.min_dist, spread=args.spread)
        fig = sc.pl.umap(adata, color=cell_type_keys,
                         show=show, return_fig=True)
        if save:
            fig.savefig(
                os.path.join(
                    ckpt_dir, f'{train_instance_name}_{emb_name}_step{step}.jpg'),
                dpi=300, bbox_inches='tight'
            )
        fig.clf()
        plt.close(fig)


def _start_shell(local_ns):
    # An interactive shell useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)
