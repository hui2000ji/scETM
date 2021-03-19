import time
import os
import logging
import pandas as pd
import scipy
import matplotlib.pyplot as plt 
import anndata
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.functional import norm

def get_train_instance_name(args):
    strs = [
        args.dataset_str,
        args.model]
    for tuple_ in (
            ('nTopics', args.n_topics, 100),
            ('lr', args.lr, 2e-2),
            ('maxKLWeight', args.max_kl_weight, 1e-6),
            ('minKLWeight', args.min_kl_weight, 1e-8),
            ('KLWeightAnneal', args.warmup_ratio, 1/3),
            ('maskRatio', args.mask_ratio, 0.),
            ('supervised', args.max_supervised_weight, 0.),
            ('trnGeneEmbDim', args.trainable_gene_emb_dim, 300)
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
            ('inputBatchID', args.input_batch_id),
            (args.log_str, args.log_str),
            ('batchScaling', args.batch_scaling),
            ('globalBias', args.global_bias),
            ('normCells', args.norm_cells),
            ('normedLoss', args.normed_loss)
    ):
        if bool_:
            strs.append(name)
    current_time = time.strftime('%m_%d-%H_%M_%S')
    strs.append('time%s' % current_time)
    train_instance_name = '_'.join(strs)
    return train_instance_name


def get_kl_weight(args, epoch):
    if args.warmup_ratio:
        kl_weight = max(min(1., epoch / (args.n_epochs * args.warmup_ratio))
                   * args.max_kl_weight, args.min_kl_weight)
    else:
        kl_weight = args.max_kl_weight
    return kl_weight


def clustering(use_rep, adata, args):
    logging.debug(f'Performing {args.clustering_method} clustering')
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep=use_rep)
    clustering_method = sc.tl.leiden if args.clustering_method == 'leiden' else sc.tl.louvain
    aris = []
    for res in args.resolutions:
        col = f'{args.clustering_method}_{res}'
        clustering_method(adata, resolution=res, key_added=col)
        ari = adjusted_rand_score(adata.obs.cell_types, adata.obs[col])
        nmi = normalized_mutual_info_score(adata.obs.cell_types, adata.obs[col])
        n_unique = adata.obs[col].nunique()
        aris.append((res, ari, n_unique))
        if 'batch_indices' in adata.obs and adata.obs.batch_indices.nunique() > 1:
            ari_batch = adjusted_rand_score(adata.obs.batch_indices, adata.obs[col])
            logging.info(f'Resolution: {res:5.3g}\tARI: {ari:7.4f}\tNMI: {nmi:7.4f}\tbARI: {ari_batch:7.4f}\t# labels: {n_unique}')
        else:
            logging.info(f'Resolution: {res:5.3g}\tARI: {ari:7.4f}\tNMI: {nmi:7.4f}\t# labels: {n_unique}')
    
    aris.sort(key=lambda x: x[1], reverse=True)
    best_res, best_ari = aris[0][0], aris[0][1]

    if not args.fix_resolutions and len(aris) > 2:
        # try to automatically adjust resolution values
        aris.sort(key=lambda x: x[2])
        n_labels = adata.obs.cell_types.nunique()
        if aris[1][2] < n_labels / 2 and aris[-1][2] <= n_labels:
            args.resolutions = [res + min(0.05, min(args.resolutions)) for res in args.resolutions]
        elif aris[-1][2] > n_labels and args.resolutions[0] > 0.01:
            args.resolutions = [res - min(0.1, min(args.resolutions) / 2) for res in args.resolutions]
    return f'{args.clustering_method}_{best_res}', best_ari


def draw_embeddings(adata: anndata.AnnData, args, color_by: list, fname: str, use_rep: str,
                    save: bool = True, show: bool = False):
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=args.n_neighbors)
    sc.tl.umap(adata, min_dist=args.min_dist, spread=args.spread)
    fig = sc.pl.umap(adata, color=color_by,
                        show=show, return_fig=True)
    if save:
        fig.savefig(
            os.path.join(args.ckpt_dir, fname),
            dpi=300, bbox_inches='tight'
        )
    fig.clf()
    plt.close(fig)


def save_embeddings(model, adata, embeddings, args):
    save_dict = dict(
        delta=embeddings['delta'],
        alpha=model.alpha.detach().cpu().numpy(),
        gene_names=adata.var_names,
        recon_log=embeddings['recon_log']
    )
    if model.rho is not None:
        save_dict['rho'] = model.rho.detach().cpu().numpy()
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
    logging.info('Calculating batch mixing entropy...')
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
    for _ in range(n_pools):
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


def initialize_logger(ckpt_dir=None):
    stream_handler = logging.StreamHandler()
    if ckpt_dir is not None:
        file_handler = logging.FileHandler(os.path.join(ckpt_dir, 'log.txt'))
    logging.basicConfig(
        handlers=[stream_handler] if ckpt_dir is None else [stream_handler, file_handler],
        format='%(levelname)s [%(asctime)s]: %(message)s',
        level=logging.INFO
    )
