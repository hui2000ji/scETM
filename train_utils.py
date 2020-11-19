import time
import os

import pandas as pd
import scipy
import matplotlib.pyplot as plt 
import anndata
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, \
    normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

def get_train_instance_name(args):
    strs = [
        args.dataset_str,
        args.model]
    for tuple_ in (
            ('nTopics', args.n_topics, 100),
            ('lr', args.lr, 2e-2),
            ('maxKLWeight', args.max_kl_weight, 1.),
            ('minKLWeight', args.min_kl_weight),
            ('KLWeightAnneal', args.kl_weight_anneal, 300),
            ('inputBatchID', args.input_batch_id),
            ('maskRatio', args.mask_ratio, 0.2),
            ('supervised', args.max_supervised_weight, 0.)
    ):
        if len(tuple_) == 2:
            name, numeric_ = tuple_
            default = 0
        else:
            name, numeric_, default = tuple_
        if numeric_ != default:
            strs.append('%s%g' % (name, numeric_))
    for name, bool_ in (
            ('qn', args.quantile_norm),
            ('log1p', args.log1p),
            ('normRdCnt', args.norm_cell_read_counts),
            (args.log_str, args.log_str),
            ('batchScaling', args.batch_scaling),
            ('normCells', args.norm_cells),
            ('normedLoss', args.normed_loss)
    ):
        if bool_:
            strs.append(name)
    current_time = time.strftime('%m_%d-%H_%M_%S')
    strs.append('time%s' % current_time)
    train_instance_name = '_'.join(strs)
    return train_instance_name


def logging(logging_items, ckpt_dir, time_str=None):
    if not isinstance(logging_items, str):
        logging_str = '; '.join(['{} {}'.format(key, val) for key, val in logging_items])
    else:
        logging_str = logging_items
    if time_str is None:
        time_str = time.strftime('%m-%d %H:%M:%S')
    logging_str = logging_str + ' ' + time_str
    with open(os.path.join(ckpt_dir, 'log.txt'), 'a+') as f:
        f.write(logging_str + '\n')
    print(logging_str, flush=True)
    return logging_str


def get_kl_weight(args, epoch):
    if args.kl_weight_anneal:
        kl_weight = max(min(1., epoch / args.kl_weight_anneal)
                   * args.max_kl_weight, args.min_kl_weight)
    else:
        kl_weight = args.max_kl_weight
    return kl_weight


def get_logging_items(embeddings, epoch, args, adata,
                      tracked_items, tracked_metric, cell_types, metadata):
    print('Evaluating and logging...', flush=True, end='\r')
    items = [('epoch', '%7d' % epoch)]
    if args.lr_decay < 1.:
        items.append(('lr', '%7.2e' % args.lr))
    for key, val in tracked_items.items():
        items.append((key, '%7.4f' % np.mean(val)))
    for cell_type_key in cell_types:
        cell_type = cell_types[cell_type_key]
        cell_type_key = cell_type_key.split('_')[0]
        tracked_metric['%s_lbl' % cell_type_key][epoch] = \
            [(cell_type == label).sum() for label in np.unique(cell_type)]
        tracked_metric['%s_nmi' % cell_type_key][epoch] = \
            normalized_mutual_info_score(adata.obs.cell_types, cell_type)
        tracked_metric['%s_ari' % cell_type_key][epoch] = \
            adjusted_rand_score(adata.obs.cell_types, cell_type)
        # tracked_metric['%s_ami' % cell_type_key][epoch] = \
        #     adjusted_mutual_info_score(adata.obs.cell_types, cell_type)
        items.extend([
            ('%s_lbl' % cell_type_key, str(
                tracked_metric['%s_lbl' % cell_type_key][epoch])),
            ('%s_nmi' % cell_type_key, '%7.4f' %
                tracked_metric['%s_nmi' % cell_type_key][epoch]),
            ('%s_ari' % cell_type_key, '%7.4f' %
                tracked_metric['%s_ari' % cell_type_key][epoch]),
            # ('%s_ami' % cell_type_key, '%7.4f' %
            #     tracked_metric['%s_ami' % cell_type_key][epoch]),
        ])
        if adata.obs.batch_indices.nunique() > 1:
            items.append((f'{cell_type_key}_bARI', '%7.4f' %
                adjusted_rand_score(adata.obs.batch_indices, cell_type)))
    if adata.obs.batch_indices.nunique() > 1 and \
            ((not args.eval and epoch == args.n_epochs) or (args.eval and epoch == args.restore_epoch)) and \
            not args.no_be:  # Only calc BE at last step
        for name, latent_space in embeddings.items():
            items.append((f'{name}_BE', '%7.4f' % 
                entropy_batch_mixing(latent_space, adata.obs.batch_indices)))
    for cluster_method, metadata_item in metadata.items():
        for k, v in metadata_item.items():
            items.append((f'{cluster_method}_{k}', '%8.5f' % v))
    # clear tracked_items    
    for key in tracked_items:
        tracked_items[key] = list()
    return items


def draw_embeddings(adata: anndata.AnnData, epoch: int, args, cell_types: dict, embeddings: dict,
                    ckpt_dir: str, save: bool = True, show: bool = False, fname_postfix: str=''):
    print('Drawing embeddings...', flush=True, end='\r')
    cell_type_keys = []
    for cell_type_key in cell_types:
        if cell_type_key.endswith('cell_type'):
            prefix = cell_type_key.split('_')[0]
            cell_type = cell_types[cell_type_key]
            if prefix in args.always_draw or epoch == args.n_epochs:
                cell_type_keys.append(prefix)
                adata.obs[prefix] = cell_type
                adata.obs[prefix] = adata.obs[prefix].astype(
                    'str').astype('category')
    if adata.obs.batch_indices.nunique() > 1:
        cell_type_keys.append('batch_indices')
    cell_type_keys.append('cell_types')
    for key in args.always_draw:
        if key != 'batch_indices' and not key in cell_type_keys and key in adata.obs:
            cell_type_keys.append(key)
            adata.obs[key] = adata.obs[key].astype(
                'str').astype('category')
    for emb_name, emb in embeddings.items():
        adata.obsm[emb_name] = emb
        sc.pp.neighbors(adata, use_rep=emb_name, n_neighbors=args.n_neighbors)
        sc.tl.umap(adata, min_dist=args.min_dist, spread=args.spread)
        fig = sc.pl.umap(adata, color=cell_type_keys,
                         show=show, return_fig=True)
        if save:
            fig.savefig(
                os.path.join(
                    ckpt_dir, f'{emb_name}_epoch{epoch}{"_" + fname_postfix if fname_postfix else ""}.pdf'),
                dpi=300, bbox_inches='tight'
            )
        fig.clf()
        plt.close(fig)


def save_embeddings(model, adata, embeddings, args):
    save_dict = dict(
        delta=embeddings['delta'],
        alpha=model.alpha.detach().cpu().numpy(),
        gene_names=adata.var_names
    )
    # if model.rho_fixed is not None:
    #     save_dict['rho_fixed'] = model.rho_fixed.detach().cpu().numpy()
    if model.rho is not None:
        save_dict['rho'] = model.rho.detach().cpu().numpy()
    if 'X_umap' in adata.obsm:
        save_dict['delta_umap'] = adata.obsm['X_umap']
    import pickle
    with open(os.path.join(args.ckpt_dir, 'embeddings.pkl'), 'wb') as f:
        pickle.dump(save_dict, f)


def _start_shell(local_ns):
    # An interactive shell useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


def entropy_batch_mixing(latent_space, batches, n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    # code adapted from scGAN
    print('Calculating batch mixing entropy...', end='\r', flush=True)
    def entropy(hist_data):
        counts = pd.Series(hist_data).value_counts()
        freqs = counts / counts.sum()
        return (-freqs * np.log(freqs + 1e-20)).sum()

    n_neighbors = min(n_neighbors, len(latent_space) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(latent_space)
    kmatrix = nne.kneighbors_graph(
        latent_space) - scipy.sparse.identity(latent_space.shape[0])

    score = 0.
    for t in range(n_pools):
        indices = np.random.choice(
            np.arange(latent_space.shape[0]), size=n_samples_per_pool)
        score += np.mean(
            [
                entropy(
                    batches[
                        kmatrix[indices[i]].nonzero()[1]
                    ]
                )
                for i in range(n_samples_per_pool)
            ]
        )
    return score / n_pools