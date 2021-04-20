from math import inf
import os
import logging
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.sparse.csr import spmatrix
from scipy.stats import chi2
from typing import Mapping, Sequence, Tuple, Iterable, Union
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from scETM.logging_utils import log_arguments
import psutil

_cpu_count: Union[None, int] = psutil.cpu_count(logical=False)
if _cpu_count is None:
    _cpu_count: int = psutil.cpu_count(logical=True)
_logger = logging.getLogger(__name__)


@log_arguments
def evaluate(adata: ad.AnnData,
    embedding_key: str = 'delta',
    n_neighbors: int = 15,
    resolutions: Iterable[float] = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
    clustering_method: str = "leiden",
    cell_type_col: str = "cell_types",
    batch_col: Union[str, None] = "batch_indices",
    draw: bool = True,
    color_by: Iterable[str] = None,
    return_fig: bool = False,
    plot_fname: str = "umap",
    plot_ftype: str = "pdf",
    plot_dir: Union[str, None] = None,
    min_dist: float = 0.3,
    spread: float = 1,
    n_jobs: int = -1,
) -> Mapping[str, Union[float, None, Figure]]:
    """Evaluates the clustering and batch correction performance of the given
    embeddings, and optionally plots the embeddings.

    WARNING: In an interactive environment, set n_jobs to 1 to avoid pickling
    error.

    Args:
        adata: the dataset with the embedding to be evaluated.
        embedding_key: the key to the embedding. Must be in adata.obsm.
        n_neighbors: #neighbors used when computing neithborhood graph and
            calculating entropy of batch mixing / kBET.
        resolutions: a sequence of resolutions used for clustering.
        clustering_method: clustering method used. Should be one of 'leiden' or
            'louvain'.
        cell_type_col: a key in adata.obs to the cell type column.
        batch_col: a key in adata.obs to the batch column.
        draw: whether to plot the embeddings.
        return_fig: whether to return the Figure object. Useful for visualizing
            the plot. Only used if draw is True.
        color_by: a list of adata.obs column keys to color the embeddings by.
            Only used if draw is True.
        plot_fname: file name of the generated plot. Only used if draw is True.
        plot_ftype: file type of the generated plot. Only used if draw is True.
        plot_dir: directory to save the generated plot. If None, do not save
            the plot. Only used if draw is True.
        min_dist: the min_dist argument in sc.tl.umap. Only used if draw is
            True.
        spread: the spread argument in sc.tl.umap. Only used if draw is True.
        n_jobs: # jobs to generate. If <= 0, this is set to the number of
            physical cores.

    Returns:
        A dict storing the ari, nmi, ebm and k_bet of the cell embeddings with
        key "ari", "nmi", "ebm", "k_bet", respectively. If draw is True and
        return_fig is True, will also store the plotted figure with key "fig".
    """

    if not pd.api.types.is_categorical_dtype(adata.obs[cell_type_col]):
        _logger.warning("scETM.evaluate assumes discrete cell types. Converting cell_type_col to categorical.")
    if not pd.api.types.is_categorical_dtype(adata.obs[batch_col]):
        _logger.warning("scETM.evaluate assumes discrete batches. Converting batch_col to categorical.")

    # calculate neighbors
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=embedding_key)

    # calculate clustering metrics
    if cell_type_col in adata.obs:
        cluster_key, best_ari, best_nmi = clustering(adata, resolutions=resolutions, cell_type_col=cell_type_col, batch_col=batch_col, clustering_method=clustering_method)
    else:
        cluster_key = best_ari = best_nmi = None

    # calculate batch correction metrics
    if batch_col and adata.obs[batch_col].nunique() > 1:
        ebm = calculate_entropy_batch_mixing(adata, use_rep=embedding_key, batch_col=batch_col, n_neighbors=n_neighbors, calc_knn=False, n_jobs=n_jobs)
        _logger.info(f'{embedding_key}_BE: {ebm:7.4f}')
        k_bet = calculate_kbet(adata, use_rep=embedding_key, batch_col=batch_col, n_neighbors=n_neighbors, calc_knn=False, n_jobs=n_jobs)[2]
        _logger.info(f'{embedding_key}_kBET: {k_bet:7.4f}')
    else:
        ebm = k_bet = None

    # plot UMAP embeddings
    if draw:
        if color_by is None:
            color_by = [cell_type_col] if batch_col is None else [batch_col, cell_type_col]
        if cluster_key is not None:
            color_by = [cluster_key] + color_by
        fig = draw_embeddings(adata=adata, color_by=color_by, min_dist=min_dist, spread=spread,
            ckpt_dir=plot_dir, fname=f'{plot_fname}.{plot_ftype}', return_fig=return_fig)
    else:
        fig = None
    
    return dict(
        ari=best_ari,
        nmi=best_nmi,
        ebm=ebm,
        k_bet=k_bet,
        fig=fig
    )


def _eff_n_jobs(n_jobs: int) -> int:
    """If n_jobs <= 0, set it as the number of physical cores _cpu_count"""
    return int(n_jobs) if n_jobs > 0 else _cpu_count


def calculate_nearest_neighbors(
    X: np.array,
    n_neighbors: int = 100,
    n_jobs: int = -1,
    method: Union[str, None] = None,
    M: int = 20,
    efC: int = 200,
    efS: int = 200,
    random_state: int = 0,
    full_speed: int = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the nearest neighbors graph for the given data.

    Args:
        X: the sample-by-feature matrix.
        n_neighbors: # nearest neighbors to find.
        n_jobs: # jobs to generate. If <= 0, this is set to the number of
            physical cores.
        method: method to calculate the NN graph. Must be "hnsw" or "sklearn".
        M: maximum number of outgoing connections in the graph. Used only if
            method is "hnsw".
        efC: parameter that controls the index_time/index_accuracy. Bigger efC
            leads to longer construction, but better index quality. At some
            point, increasing efC does not improve the quality of the index.
            Used only if method is "hnsw".
        efS: the size of the dynamic list for the nearest neighbors (used
            during the search). Higher efS leads to more accurate but slower
            search. Can be anything between n_neighbors and #samples. Used only
            if method is "hnsw".
        random_state: random seed. Used only if method is "hnsw".
        full_speed: If full_speed, use multiple threads in constructing hnsw
            index. However, the kNN results are not reproducible. Used only if
            method is "hnsw".

    Returns:
        indicies: indices of the (n_neighbors - 1)-nearest neighbors, stored as
            a np.ndarray.
        distances: distances of the (n_neighbors - 1)-nearest neighbors, stored
            as a np.ndarray.
    """

    nsamples = X.shape[0]

    if method is None:
        if nsamples <= 1000:
            method = "sklearn"
        else:
            method = "hnsw"

    if nsamples < n_neighbors:
        _logger.warning(f"Warning: in calculate_nearest_neighbors, number of samples = {nsamples} < n_neighbors = {n_neighbors}!\n Set n_neighbors to {nsamples}.")
        n_neighbors = nsamples

    n_jobs = _eff_n_jobs(n_jobs)

    if method == "hnsw":
        import hnswlib

        if issparse(X):
            X = X.toarray()
        # Build hnsw index
        knn_index = hnswlib.Index(space="l2", dim=X.shape[1])
        knn_index.init_index(
            max_elements=nsamples, ef_construction=efC, M=M, random_seed=random_state
        )
        knn_index.set_num_threads(n_jobs if full_speed else 1)
        knn_index.add_items(X)

        # KNN query
        knn_index.set_ef(efS)
        knn_index.set_num_threads(n_jobs)
        indices, distances = knn_index.knn_query(X, k=n_neighbors)
        # eliminate the first neighbor, which is the node itself
        if not (indices[:, 0] == np.arange(nsamples)).all():
            for i in range(nsamples):
                if indices[i, 0] != i:
                    indices[i, 1:] = indices[i, 0:-1]
                    distances[i, 1:] = distances[i, 0:-1]
        indices = indices[:, 1:].astype(int)
        distances = np.sqrt(distances[:, 1:])
    else:
        assert method == "sklearn"
        knn = NearestNeighbors(
            n_neighbors=n_neighbors - 1, n_jobs=n_jobs
        )  # eliminate the first neighbor, which is the node itself
        knn.fit(X)
        distances, indices = knn.kneighbors()

    return indices, distances


def _calculate_kbet_for_one_chunk(knn_indices, attr_values, ideal_dist, n_neighbors):
    dof = ideal_dist.size - 1

    ns = knn_indices.shape[0]
    results = np.zeros((ns, 2))
    for i in range(ns):
        # NOTE: Do not use np.unique. Some of the batches may not be present in
        # the neighborhood.
        observed_counts = pd.Series(attr_values[knn_indices[i, :]]).value_counts(sort=False).values
        expected_counts = ideal_dist * n_neighbors
        stat = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
        p_value = 1 - chi2.cdf(stat, dof)
        results[i, 0] = stat
        results[i, 1] = p_value

    return results


def _get_knn_indices(adata: ad.AnnData,
    use_rep: str = "delta",
    n_neighbors: int = 25,
    random_state: int = 0,
    full_speed: bool = False,
    n_jobs: int = -1,
    calc_knn: bool = True
) -> np.ndarray:

    nsample = adata.n_obs
    if calc_knn:
        assert use_rep in adata.obsm, f'{use_rep} not in adata.obsm'
        indices, distances = calculate_nearest_neighbors(
            adata.obsm[use_rep],
            n_neighbors=n_neighbors,
            n_jobs=_eff_n_jobs(n_jobs),
            random_state=random_state,
            full_speed=full_speed,
        )
    else:
        assert 'neighbors' in adata.uns, 'No precomputed knn exists.'
        assert adata.uns['neighbors']['params']['n_neighbors'] >= n_neighbors, f"pre-computed n_neighbors is {adata.uns['neighbors']['params']['n_neighbors']}, which is smaller than {n_neighbors}"
        indices = adata.obsp['distances'].nonzero()[1].reshape(adata.n_obs, -1)
        indices = indices[:, :n_neighbors - 1]
    knn_indices = np.concatenate(
        (np.arange(nsample).reshape(-1, 1), indices[:, :n_neighbors - 1]), axis=1
    )  # add query as 1-nn
    return knn_indices


def calculate_kbet(
    adata: ad.AnnData,
    use_rep: str = "delta",
    batch_col: str = "batch_indices",
    n_neighbors: int = 25,
    alpha: float = 0.05,
    random_state: int = 0,
    full_speed: bool = False,
    n_jobs: int = -1,
    calc_knn: bool = True
) -> Tuple[float, float, float]:
    """Calculates the kBET metric of the data.

    kBET measures if cells from different batches mix well in their local
    neighborhood.

    Args:
        adata: annotated data matrix.
        use_rep: the embedding to be used. Must exist in adata.obsm.
        batch_col: a key in adata.obs to the batch column.
        n_neighbors: # nearest neighbors.
        alpha: acceptance rate threshold. A cell is accepted if its kBET
            p-value is greater than or equal to alpha.
        random_state: random seed. Used only if method is "hnsw".
        full_speed: If full_speed, use multiple threads in constructing hnsw
            index. However, the kNN results are not reproducible. Used only if
            method is "hnsw".
        n_jobs: # jobs to generate. If <= 0, this is set to the number of
            physical cores.
        calc_knn: whether to re-calculate the kNN graph or reuse the one stored
            in adata.

    Returns:
        stat_mean: mean kBET chi-square statistic over all cells.
        pvalue_mean: mean kBET p-value over all cells.
        accept_rate: kBET Acceptance rate of the sample.
    """

    _logger.info('Calculating kbet...')
    assert batch_col in adata.obs
    if adata.obs[batch_col].dtype.name != "category":
        adata.obs[batch_col] = adata.obs[batch_col].astype('category')

    ideal_dist = (
        adata.obs[batch_col].value_counts(normalize=True, sort=False).values
    )  # ideal no batch effect distribution
    nsample = adata.shape[0]
    nbatch = ideal_dist.size

    attr_values = adata.obs[batch_col].values.copy()
    attr_values.categories = range(nbatch)
    knn_indices = _get_knn_indices(adata, use_rep, n_neighbors, random_state, full_speed, n_jobs, calc_knn)

    # partition into chunks
    n_jobs = min(_eff_n_jobs(n_jobs), nsample)
    starts = np.zeros(n_jobs + 1, dtype=int)
    quotient = nsample // n_jobs
    remainder = nsample % n_jobs
    for i in range(n_jobs):
        starts[i + 1] = starts[i] + quotient + (1 if i < remainder else 0)

    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("loky", n_jobs=n_jobs):
        kBET_arr = np.concatenate(
            Parallel()(
                delayed(_calculate_kbet_for_one_chunk)(
                    knn_indices[starts[i] : starts[i + 1], :], attr_values, ideal_dist, n_neighbors
                )
                for i in range(n_jobs)
            )
        )

    res = kBET_arr.mean(axis=0)
    stat_mean = res[0]
    pvalue_mean = res[1]
    accept_rate = (kBET_arr[:, 1] >= alpha).sum() / nsample

    return (stat_mean, pvalue_mean, accept_rate)


def _entropy(hist_data):
    _, counts = np.unique(hist_data, return_counts = True)
    freqs = counts / counts.sum()
    return (-freqs * np.log(freqs + 1e-30)).sum()


def _entropy_batch_mixing_for_one_pool(batches, knn_indices, nsample, n_samples_per_pool):
    indices = np.random.choice(
        np.arange(nsample), size=n_samples_per_pool)
    return np.mean(
        [
            _entropy(batches[knn_indices[indices[i]]])
            for i in range(n_samples_per_pool)
        ]
    )


def calculate_entropy_batch_mixing(
    adata: ad.AnnData,
    use_rep: str = "delta",
    batch_col: str = "batch_indices",
    n_neighbors: int = 50,
    n_pools: int = 50,
    n_samples_per_pool: int = 100,
    random_state: int = 0,
    full_speed: bool = False,
    n_jobs: int = -1,
    calc_knn: bool = True
) -> float:
    """Calculates the entropy of batch mixing of the data.

    kBET measures if cells from different batches mix well in their local
    neighborhood.

    Args:
        adata: annotated data matrix.
        use_rep: the embedding to be used. Must exist in adata.obsm.
        batch_col: a key in adata.obs to the batch column.
        n_neighbors: # nearest neighbors.
        n_pools: #pools of cells to calculate entropy of batch mixing.
        n_samples_per_pool: #cells per pool to calculate within-pool entropy.
        random_state: random seed. Used only if method is "hnsw".
        full_speed: If full_speed, use multiple threads in constructing hnsw
            index. However, the kNN results are not reproducible. Used only if
            method is "hnsw".
        n_jobs: # jobs to generate. If <= 0, this is set to the number of
            physical cores.
        calc_knn: whether to re-calculate the kNN graph or reuse the one stored
            in adata.

    Returns:
        score: the mean entropy of batch mixing, averaged from n_pools samples.
    """

    _logger.info('Calculating batch mixing entropy...')
    nsample = adata.n_obs

    knn_indices = _get_knn_indices(adata, use_rep, n_neighbors, random_state, full_speed, n_jobs, calc_knn)

    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("loky", inner_max_num_threads=1):
        score = np.mean(
            Parallel(n_jobs=n_jobs)(
                delayed(_entropy_batch_mixing_for_one_pool)(
                    adata.obs[batch_col], knn_indices, nsample, n_samples_per_pool
                )
                for _ in range(n_pools)
            )
        )
    return score


def clustering(
    adata: ad.AnnData,
    resolutions: Sequence[float],
    clustering_method: str = "leiden",
    cell_type_col: str = "cell_types",
    batch_col: str = "batch_indices"
) -> Tuple[str, float, float]:
    """Clusters the data and calculate agreement with cell type and batch
    variable.

    This method cluster the neighborhood graph (requires having run sc.pp.
    neighbors first) with "clustering_method" algorithm multiple times with the
    given resolutions, and return the best result in terms of ARI with cell
    type.
    Other metrics such as NMI with cell type, ARi with batch are logged but not
    returned. (TODO: also return these metrics)

    Args:
        adata: the dataset to be clustered. adata.obsp shouhld contain the keys
            'connectivities' and 'distances'.
        resolutions: a list of leiden/louvain resolution parameters. Will
            cluster with each resolution in the list and return the best result
            (in terms of ARI with cell type).
        clustering_method: Either "leiden" or "louvain".
        cell_type_col: a key in adata.obs to the cell type column.
        batch_col: a key in adata.obs to the batch column.

    Returns:
        best_cluster_key: a key in adata.obs to the best (in terms of ARI with
            cell type) cluster assignment column.
        best_ari: the best ARI with cell type.
        best_nmi: the best NMI with cell type.
    """

    assert len(resolutions) > 0, f'Must specify at least one resolution.'

    if clustering_method == 'leiden':
        clustering_func: function = sc.tl.leiden
    elif clustering_method == 'louvain':
        clustering_func: function = sc.tl.louvain
    else:
        raise ValueError("Please specify louvain or leiden for the clustering method argument.")
    _logger.info(f'Performing {clustering_method} clustering')
    assert cell_type_col in adata.obs, f"{cell_type_col} not in adata.obs"
    best_res, best_ari, best_nmi = None, -inf, -inf
    for res in resolutions:
        col = f'{clustering_method}_{res}'
        clustering_func(adata, resolution=res, key_added=col)
        ari = adjusted_rand_score(adata.obs[cell_type_col], adata.obs[col])
        nmi = normalized_mutual_info_score(adata.obs[cell_type_col], adata.obs[col])
        n_unique = adata.obs[col].nunique()
        if ari > best_ari:
            best_res = res
            best_ari = ari
        if nmi > best_nmi:
            best_nmi = nmi
        if batch_col in adata.obs and adata.obs[batch_col].nunique() > 1:
            ari_batch = adjusted_rand_score(adata.obs[batch_col], adata.obs[col])
            _logger.info(f'Resolution: {res:5.3g}\tARI: {ari:7.4f}\tNMI: {nmi:7.4f}\tbARI: {ari_batch:7.4f}\t# labels: {n_unique}')
        else:
            _logger.info(f'Resolution: {res:5.3g}\tARI: {ari:7.4f}\tNMI: {nmi:7.4f}\t# labels: {n_unique}')
    
    return f'{clustering_method}_{best_res}', best_ari, best_nmi


def draw_embeddings(adata: ad.AnnData,
        color_by: Union[str, Sequence[str], None] = None,
        min_dist: float = 0.3,
        spread: float = 1,
        ckpt_dir: str = '.',
        fname: str = "umap.pdf",
        return_fig: bool = False,
        dpi: int = 300
    ) -> Union[None, Figure]:
    """Embeds, plots and optionally saves the neighborhood graph with UMAP.

    Requires having run sc.pp.neighbors first.

    Args:
        adata: the dataset to draw. adata.obsp shouhld contain the keys
            'connectivities' and 'distances'.
        color_by: a str or a list of adata.obs keys to color the points in the
            scatterplot by. E.g. if both cell_type_col and batch_col is in
            color_by, then we would have two plots colored by cell type and
            batch variables, respectively.
        min_dist: The effective minimum distance between embedded points.
            Smaller values will result in a more clustered/clumped embedding
            where nearby points on the manifold are drawn closer together,
            while larger values will result on a more even dispersal of points.
        spread: The effective scale of embedded points. In combination with
            `min_dist` this determines how clustered/clumped the embedded
            points are.
        ckpt_dir: where to save the plot. If None, do not save the plot.
        fname: file name of the saved plot. Only used if ckpt_dir is not None.
        return_fig: whether to return the Figure object. Useful for visualizing
            the plot.
        dpi: the dpi of the saved plot. Only used if ckpt_dir is not None.

    Returns:
        If return_fig is True, return the figure containing the plot.
    """

    sc.tl.umap(adata, min_dist=min_dist, spread=spread)
    fig = sc.pl.umap(adata, color=color_by, show=False, return_fig=True)
    if ckpt_dir is not None:
        assert os.path.exists(ckpt_dir), f'ckpt_dir {ckpt_dir} does not exist.'
        fig.savefig(
            os.path.join(ckpt_dir, fname),
            dpi=dpi, bbox_inches='tight'
        )
    if return_fig:
        return fig
    fig.clf()
    plt.close(fig)
