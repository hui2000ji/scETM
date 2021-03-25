import pickle
import logging
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import chi2
from typing import Tuple, List
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
import psutil
_cpu_count = psutil.cpu_count(logical=False)
if _cpu_count is None:
    _cpu_count = psutil.cpu_count(logical=True)


def eff_n_jobs(n_jobs: int) -> int:
    """ If n_jobs < 0, set it as the number of physical cores _cpu_count """
    return n_jobs if n_jobs > 0 else _cpu_count


def calculate_kbet_from_path(emb_path, data_path):
    f = open(emb_path, 'rb')
    emb = pickle.load(f)
    adata = ad.read_h5ad(data_path)
    adata.obsm['X_latent']=emb
    stat_mean, pvalue_mean, accept_rate = calc_kBET(adata, 'batch_indices',
        rep='X_latent',
        K=25,
        alpha=0.05
    )
    return (stat_mean, pvalue_mean, accept_rate)
  

def calculate_kbet_from_adata(adata, rep='X_latent', K=25, alpha=0.05, random_state=0, full_speed=True):
    '''adata object needs to have .obsm['X_latent'] and .obs['batch_indices']'''
    stat_mean, pvalue_mean, accept_rate = calc_kBET(adata, 'batch_indices',
        rep=rep,
        K=K,
        alpha=alpha,
        random_state=random_state,
        full_speed=full_speed
    )
    return (stat_mean, pvalue_mean, accept_rate)


def X_from_rep(data: ad.AnnData, rep: str) -> np.array:
    """
    If rep is not X, first check if X_rep is in data.obsm. If not, raise an error.
    If rep is None, return data.X as a numpy array
    """
    if rep != "X":
        if rep not in data.obsm.keys():
            raise ValueError("Cannot find {0} matrix. Please run {0} first".format(rep))
        return data.obsm[rep]
    else:
        return data.X if not issparse(data.X) else data.X.toarray()


def calculate_nearest_neighbors(
    X: np.array,
    K: int = 100,
    n_jobs: int = -1,
    method: str = "hnsw",
    M: int = 20,
    efC: int = 200,
    efS: int = 200,
    random_state: int = 0,
    full_speed: int = False,
):
    """Calculate nearest neighbors
    X is the sample by feature matrix
    Return K -1 neighbors, the first one is the point itself and thus omitted.
    TODO: Documentation
    """

    nsample = X.shape[0]

    if nsample <= 1000:
        method = "sklearn"

    if nsample < K:
        logging.warning(f"Warning: in calculate_nearest_neighbors, number of samples = {nsample} < K = {K}!\n Set K to {nsample}.")
        K = nsample

    n_jobs = eff_n_jobs(n_jobs)

    if method == "hnsw":
        import hnswlib

        if issparse(X):
            X = X.toarray()
        # Build hnsw index
        knn_index = hnswlib.Index(space="l2", dim=X.shape[1])
        knn_index.init_index(
            max_elements=nsample, ef_construction=efC, M=M, random_seed=random_state
        )
        knn_index.set_num_threads(n_jobs if full_speed else 1)
        knn_index.add_items(X)

        # KNN query
        knn_index.set_ef(efS)
        knn_index.set_num_threads(n_jobs)
        indices, distances = knn_index.knn_query(X, k=K)
        # eliminate the first neighbor, which is the node itself
        if not (indices[:, 0] == np.arange(nsample)).all():
            for i in range(nsample):
                if indices[i, 0] != i:
                    indices[i, 1:] = indices[i, 0:-1]
                    distances[i, 1:] = distances[i, 0:-1]
        indices = indices[:, 1:].astype(int)
        distances = np.sqrt(distances[:, 1:])
    else:
        assert method == "sklearn"
        knn = NearestNeighbors(
            n_neighbors=K - 1, n_jobs=n_jobs
        )  # eliminate the first neighbor, which is the node itself
        knn.fit(X)
        distances, indices = knn.kneighbors()

    return indices, distances


def calc_kBET_for_one_chunk(knn_indices, attr_values, ideal_dist, K):
    dof = ideal_dist.size - 1

    ns = knn_indices.shape[0]
    results = np.zeros((ns, 2))
    for i in range(ns):
        observed_counts = (
            pd.Series(attr_values[knn_indices[i, :]]).value_counts(sort=False).values
        )
        expected_counts = ideal_dist * K
        stat = np.sum(
            np.divide(
                np.square(np.subtract(observed_counts, expected_counts)),
                expected_counts,
            )
        )
        p_value = 1 - chi2.cdf(stat, dof)
        results[i, 0] = stat
        results[i, 1] = p_value

    return results


def calc_kBET(
    data,
    attr: str,
    rep: str = "pca",
    K: int = 25,
    alpha: float = 0.05,
    n_jobs: int = -1,
    random_state: int = 0,
    temp_folder: str = None,
    full_speed: bool = False
) -> Tuple[float, float, float]:
    """Calculate the kBET metric of the data regarding a specific sample attribute and embedding.

    The kBET metric is defined in [Büttner18]_, which measures if cells from different samples mix well in their local neighborhood.

    Parameters
    ----------
    data: ``pegasusio.MultimodalData``
        Annotated data matrix with rows for cells and columns for genes.

    attr: ``str``
        The sample attribute to consider. Must exist in ``data.obs``.

    rep: ``str``, optional, default: ``"pca"``
        The embedding representation to be used. The key ``'X_' + rep`` must exist in ``data.obsm``. By default, use PCA coordinates.

    K: ``int``, optional, default: ``25``
        Number of nearest neighbors, using L2 metric.

    alpha: ``float``, optional, default: ``0.05``
        Acceptance rate threshold. A cell is accepted if its kBET p-value is greater than or equal to ``alpha``.

    n_jobs: ``int``, optional, default: ``-1``
        Number of threads used. If ``-1``, use all physical CPU cores.

    random_state: ``int``, optional, default: ``0``
        Random seed set for reproducing results.

    temp_folder: ``str``, optional, default: ``None``
        Temporary folder for joblib execution.

    full_speed: `bool`, optional (default: False)
        If full_speed, use multiple threads in constructing hnsw index. However, the kNN results are not reproducible. If not full_speed, use only one thread to make sure results are reproducible.

    Returns
    -------
    stat_mean: ``float``
        Mean kBET chi-square statistic over all cells.

    pvalue_mean: ``float``
        Mean kBET p-value over all cells.

    accept_rate: ``float``
        kBET Acceptance rate of the sample.

    Examples
    --------
    >>> pg.calc_kBET(data, attr = 'Channel')

    >>> pg.calc_kBET(data, attr = 'Channel', rep = 'umap')
    """
    assert attr in data.obs
    assert data.obs[attr].dtype.name == "category"

    ideal_dist = (
        data.obs[attr].value_counts(normalize=True, sort=False).values
    )  # ideal no batch effect distribution
    nsample = data.shape[0]
    nbatch = ideal_dist.size

    attr_values = data.obs[attr].values.copy()
    attr_values.categories = range(nbatch)

    indices, distances = calculate_nearest_neighbors(
        X_from_rep(data, rep),
        K=K,
        n_jobs=eff_n_jobs(n_jobs),
        random_state=random_state,
        full_speed=full_speed,
    )
    indices_key = rep + "_knn_indices"
    distances_key = rep + "_knn_distances"
    data.uns[indices_key] = indices
    data.uns[distances_key] = distances
    knn_indices = np.concatenate(
        (np.arange(nsample).reshape(-1, 1), indices[:, 0 : K - 1]), axis=1
    )  # add query as 1-nn

    # partition into chunks
    n_jobs = min(eff_n_jobs(n_jobs), nsample)
    starts = np.zeros(n_jobs + 1, dtype=int)
    quotient = nsample // n_jobs
    remainder = nsample % n_jobs
    for i in range(n_jobs):
        starts[i + 1] = starts[i] + quotient + (1 if i < remainder else 0)

    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("loky", inner_max_num_threads=1):
        kBET_arr = np.concatenate(
            Parallel(n_jobs=n_jobs, temp_folder=temp_folder)(
                delayed(calc_kBET_for_one_chunk)(
                    knn_indices[starts[i] : starts[i + 1], :], attr_values, ideal_dist, K
                )
                for i in range(n_jobs)
            )
        )

    res = kBET_arr.mean(axis=0)
    stat_mean = res[0]
    pvalue_mean = res[1]
    accept_rate = (kBET_arr[:, 1] >= alpha).sum() / nsample

    return (stat_mean, pvalue_mean, accept_rate)
