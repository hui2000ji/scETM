import pickle
import os
import numpy as np
import torch.nn.functional as F 
import torch
import pandas as pd
import anndata as ad
from scETM import scETM
from multiprocessing import Pool
from itertools import repeat


def simulate_mean_diff_once(data):
    half = len(data) // 2
    ind = np.arange(len(data))
    np.random.shuffle(ind)
    md = data[ind[:half]].mean(0) - data[ind[half:half * 2]].mean(0)
    return md


def simulate_mean_diff(data, repeats):
    mds = []
    for _ in range(repeats):
        mds.append(simulate_mean_diff_once(data))
    return mds


def calc_md_pval(series, delta_kept, mds_simulated):
    mds = []
    unique = series.unique()
    for t in unique:
        test = delta_kept[series == t]  # (cells_test, topics)
        ctrl = delta_kept[series != t]  # (cells_ctrl, topics)
        md = test.mean(0) - ctrl.mean(0)  # (topics)
        mds.append(md)
    mds = np.array(mds)  # (cell_types, topics)
    mds_simulated = np.array(mds_simulated)

    pvals = (mds_simulated.T[None, ...] > mds[..., None]).sum(-1) + 1 / (reps + 1)  # (cell_types, topics, *repeats*)
    pvals = pd.DataFrame(pvals, index=unique, columns=topic_kept)  # (cell_types, topics)
    pvals = pvals * 100 * len(unique)
    mds = pd.DataFrame(mds, index=unique, columns=topic_kept)
    return mds, pvals


if __name__ == '__main__':
    working_dir = 'AD'
    adata = ad.read_h5ad('../../../../data/AD/AD.h5ad')
    model = scETM(adata.n_vars, adata.obs.batch_indices.nunique())
    model.load_state_dict(torch.load('AD/model-1200'))
    model.get_all_embeddings_and_nll(adata)

    delta, alpha, rho = map(pd.DataFrame, [adata.obsm['delta'], adata.uns['alpha'], adata.varm['rho']])
    delta.index = adata.obs_names
    rho.index = adata.var_names
    delta.shape, alpha.shape, rho.shape

    # sample 
    delta_sampled = delta.sample(10000)
    topic_kept = delta_sampled.columns[delta_sampled.abs().sum(0) >= 1500]

    print('Starting permutation test for cell types')
    from time import time
    start = time()

    delta_kept = delta[topic_kept]  # (cells, topics)

    reps = 10000
    n_jobs = 10
    with Pool(n_jobs) as p:
        l = []
        rep_arr = [reps // n_jobs] * n_jobs
        rep_arr[-1] += reps % n_jobs
        for rep in rep_arr:
            l.append(p.apply_async(simulate_mean_diff, (delta_kept.values, rep)))
        l = [e.get() for e in l]
        mds_simulated = np.concatenate(l, axis=0)
    print(time() - start)

    mds, pvals = calc_md_pval(adata.obs.cell_types, delta_kept, mds_simulated)
    pvals.to_csv(os.path.join(working_dir, 'perm_p_onesided_celltype.csv'))
    mds.to_csv(os.path.join(working_dir, 'perm_md_onesided_celltype.csv'))

    mds, pvals = calc_md_pval(adata.obs.condition, delta_kept, mds_simulated)
    pvals.to_csv(os.path.join(working_dir, 'perm_p_onesided_condition.csv'))
    mds.to_csv(os.path.join(working_dir, 'perm_md_onesided_condition.csv'))

