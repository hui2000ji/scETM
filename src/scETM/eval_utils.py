from math import inf
import os
import logging
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.sparse.csr import spmatrix
from scipy.stats import chi2
from typing import Mapping, Sequence, Tuple, Iterable, Union
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples
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
    color_by: Iterable[str] = None,
    return_fig: bool = False,
    plot_fname: str = "umap",
    plot_ftype: str = "pdf",
    plot_dir: Union[str, None] = None,
    plot_dpi: int = 300,
    writer: Union[None, SummaryWriter] = None,
    min_dist: float = 0.3,
    spread: float = 1,
    n_jobs: int = 1,
    random_state: Union[None, int, np.random.RandomState, np.random.Generator] = 0,
    umap_kwargs: dict = dict()
) -> Mapping[str, Union[float, None, Figure]]:
    """Evaluates the clustering and batch correction performance of the given
    embeddings, and optionally plots the embeddings.

    Embeddings will be plotted if return_fig is True or plot_dir is provided.
    When tensorboard_dir is provided, will also save the embeddings using a
    tensorboard SummaryWriter.

    NOTE: Set n_jobs to 1 if you encounter pickling error.

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
        return_fig: whether to return the Figure object. Useful for visualizing
            the plot.
        color_by: a list of adata.obs column keys to color the embeddings by.
            If None, will look up adata.uns['color_by']. Only used if is
            drawing.
        plot_fname: file name of the generated plot. Only used if is drawing.
        plot_ftype: file type of the generated plot. Only used if is drawing.
        plot_dir: directory to save the generated plot. If None, do not save
            the plot.
        plot_dpi: dpi to save the plot.
        writer: an initialized SummaryWriter to save the UMAP plot to. Only
            used if is drawing.
        min_dist: the min_dist argument in sc.tl.umap. Only used is drawing.
        spread: the spread argument in sc.tl.umap. Only used if is drawing.
        n_jobs: # jobs to generate. If <= 0, this is set to the number of
            physical cores.
        random_state: random state for knn calculation.
        umap_kwargs: other kwargs to pass to sc.pl.umap.

    Returns:
        A dict storing the ari, nmi, asw, ebm and k_bet of the cell embeddings
        with key "ari", "nmi", "asw", "ebm", "k_bet", respectively. If draw is
        True and return_fig is True, will also store the plotted figure with
        key "fig".
    """

    if cell_type_col and not pd.api.types.is_categorical_dtype(adata.obs[cell_type_col]):
        _logger.warning("scETM.evaluate assumes discrete cell types. Converting cell_type_col to categorical.")
        adata.obs[cell_type_col] = adata.obs[cell_type_col].astype(str).astype('category')
    if batch_col and not pd.api.types.is_categorical_dtype(adata.obs[batch_col]):
        _logger.warning("scETM.evaluate assumes discrete batches. Converting batch_col to categorical.")
        adata.obs[batch_col] = adata.obs[batch_col].astype(str).astype('category')

    # calculate neighbors
    _get_knn_indices(adata, use_rep=embedding_key, n_neighbors=n_neighbors, random_state=random_state, calc_knn=True)

    # calculate clustering metrics
    if cell_type_col in adata.obs and len(resolutions) > 0 and adata.obs[cell_type_col].nunique() > 1:
        cluster_key, best_ari, best_nmi = clustering(adata, resolutions=resolutions, cell_type_col=cell_type_col, batch_col=batch_col, clustering_method=clustering_method)
        sw = silhouette_samples(adata.X if embedding_key == 'X' else adata.obsm[embedding_key], adata.obs[cell_type_col])
        adata.obs['silhouette_width'] = sw
        asw = np.mean(sw)
        _logger.info(f'{embedding_key}_ASW: {asw:7.4f}')
        if batch_col and cell_type_col:
            sw_table = adata.obs.pivot_table(index=cell_type_col, columns=batch_col, values="silhouette_width", aggfunc="mean")
            _logger.info(f'SW: {sw_table}')
            if plot_dir is not None:
                sw_table.to_csv(os.path.join(plot_dir, f'{plot_fname}.csv'))
    else:
        asw = cluster_key = best_ari = best_nmi = None

    # calculate batch correction metrics
    need_batch = batch_col and adata.obs[batch_col].nunique() > 1
    if need_batch:
        ebm = calculate_entropy_batch_mixing(adata,
            use_rep=embedding_key,
            batch_col=batch_col,
            n_neighbors=n_neighbors,
            calc_knn=False,
            n_jobs=n_jobs,
        )
        _logger.info(f'{embedding_key}_BE: {ebm:7.4f}')
        k_bet = calculate_kbet(adata,
            use_rep=embedding_key,
            batch_col=batch_col,
            n_neighbors=n_neighbors,
            calc_knn=False,
            n_jobs=n_jobs,
        )[2]
        _logger.info(f'{embedding_key}_kBET: {k_bet:7.4f}')
    else:
        ebm = k_bet = None

    # plot UMAP embeddings
    if return_fig or plot_dir is not None:
        if color_by is None:
            color_by = [batch_col, cell_type_col] if need_batch else [cell_type_col]
        color_by = list(color_by)
        if 'color_by' in adata.uns:
            for col in adata.uns['color_by']:
                if col not in color_by:
                    color_by.insert(0, col)
        if cluster_key is not None:
            color_by = [cluster_key] + color_by
        fig = draw_embeddings(adata=adata, color_by=color_by,
            min_dist=min_dist, spread=spread,
            ckpt_dir=plot_dir, fname=f'{plot_fname}.{plot_ftype}', return_fig=return_fig,
            dpi=plot_dpi,
            umap_kwargs=umap_kwargs)
        if writer is not None:
            writer.add_embedding(adata.obsm['X_umap'], tag=plot_fname)
    else:
        fig = None
    
    return dict(
        ari=best_ari,
        nmi=best_nmi,
        asw=asw,
        ebm=ebm,
        k_bet=k_bet,
        fig=fig
    )


def _eff_n_jobs(n_jobs: Union[None, int]) -> int:
    """If n_jobs <= 0, set it as the number of physical cores _cpu_count"""
    if n_jobs is None:
        return 1
    return int(n_jobs) if n_jobs > 0 else _cpu_count


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
    calc_knn: bool = True
) -> np.ndarray:

    if calc_knn:
        assert use_rep == 'X' or use_rep in adata.obsm, f'{use_rep} not in adata.obsm and is not "X"'
        neighbors = sc.Neighbors(adata)
        neighbors.compute_neighbors(n_neighbors=n_neighbors, knn=True, use_rep=use_rep, random_state=random_state, write_knn_indices=True)
        adata.obsp['distances'] = neighbors.distances
        adata.obsp['connectivities'] = neighbors.connectivities
        adata.obsm['knn_indices'] = neighbors.knn_indices
        adata.uns['neighbors'] = {
            'connectivities_key': 'connectivities',
            'distances_key': 'distances',
            'knn_indices_key': 'knn_indices',
            'params': {
                'n_neighbors': n_neighbors,
                'use_rep': use_rep,
                'metric': 'euclidean',
                'method': 'umap'
            }
        }
    else:
        assert 'neighbors' in adata.uns, 'No precomputed knn exists.'
        assert adata.uns['neighbors']['params']['n_neighbors'] >= n_neighbors, f"pre-computed n_neighbors is {adata.uns['neighbors']['params']['n_neighbors']}, which is smaller than {n_neighbors}"

    return adata.obsm['knn_indices']


def calculate_kbet(
    adata: ad.AnnData,
    use_rep: str = "delta",
    batch_col: str = "batch_indices",
    n_neighbors: int = 25,
    alpha: float = 0.05,
    random_state: int = 0,
    n_jobs: Union[None, int] = None,
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
        _logger.warning(f'Making the column {batch_col} of adata.obs categorical.')
        adata.obs[batch_col] = adata.obs[batch_col].astype('category')

    ideal_dist = (
        adata.obs[batch_col].value_counts(normalize=True, sort=False).values
    )  # ideal no batch effect distribution
    nsample = adata.shape[0]
    nbatch = ideal_dist.size

    attr_values = adata.obs[batch_col].values.copy()
    attr_values.categories = range(nbatch)
    knn_indices = _get_knn_indices(adata, use_rep, n_neighbors, random_state, calc_knn)

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
    n_jobs: Union[None, int] = None,
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
        n_jobs: # jobs to generate. If <= 0, this is set to the number of
            physical cores.
        calc_knn: whether to re-calculate the kNN graph or reuse the one stored
            in adata.

    Returns:
        score: the mean entropy of batch mixing, averaged from n_pools samples.
    """

    _logger.info('Calculating batch mixing entropy...')
    nsample = adata.n_obs

    knn_indices = _get_knn_indices(adata, use_rep, n_neighbors, random_state, calc_knn)

    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("loky", n_jobs=n_jobs, inner_max_num_threads=1):
        score = np.mean(
            Parallel()(
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
        dpi: int = 300,
        umap_kwargs: dict = dict()
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
        umap_kwargs: other kwargs to pass to sc.pl.umap.

    Returns:
        If return_fig is True, return the figure containing the plot.
    """

    _logger.info(f'Plotting UMAP embeddings...')
    sc.tl.umap(adata, min_dist=min_dist, spread=spread)
    fig = sc.pl.umap(adata, color=color_by, show=False, return_fig=True, **umap_kwargs)
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


def set_figure_params(
    matplotlib_backend: str = 'agg',
    dpi: int = 120,
    frameon: bool = True,
    vector_friendly: bool = True,
    fontsize: int = 10,
    figsize: Sequence[int] = (10, 10)
):
    """Set figure parameters.
    Args
        backend: the backend to switch to.  This can either be one of th
            standard backend names, which are case-insensitive:
            - interactive backends:
                GTK3Agg, GTK3Cairo, MacOSX, nbAgg,
                Qt4Agg, Qt4Cairo, Qt5Agg, Qt5Cairo,
                TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo
            - non-interactive backends:
                agg, cairo, pdf, pgf, ps, svg, template
            or a string of the form: ``module://my.module.name``.
        dpi: resolution of rendered figures – this influences the size of
            figures in notebooks.
        frameon: add frames and axes labels to scatter plots.
        vector_friendly: plot scatter plots using `png` backend even when
            exporting as `pdf` or `svg`.
        fontsize: the fontsize for several `rcParams` entries.
        figsize: plt.rcParams['figure.figsize'].
    """
    matplotlib.use(matplotlib_backend)
    sc.set_figure_params(dpi=dpi, figsize=figsize, fontsize=fontsize, frameon=frameon, vector_friendly=vector_friendly)
