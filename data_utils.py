import os
import random

import numpy as np
import pandas as pd
import scvi.dataset
from scipy.io import mmread
from tqdm import tqdm


def load_mat(root, generate_df=False):
    # read gene names
    with open(os.path.join(root, 'gene_names.txt')) as row_file:
        genes = row_file.read().splitlines()
        print("Number of genes:", len(genes))
    # read cell IDs
    with open(os.path.join(root, 'cell_ids.txt')) as col_file:
        cells = col_file.read().splitlines()
        print("Number of cells:", len(cells))
    # read count matrix
    mat = (mmread(root + 'matrix.mtx'))
    # sparse matrix to dense
    mat_dense = mat.todense()
    # transpose
    if mat_dense.shape[0] == len(genes):
        mat_dense = mat_dense.T
    print(mat_dense.shape)
    # create DataFrame
    if generate_df:
        df = pd.DataFrame(mat_dense, cells, genes)
        return df
    return mat_dense


def export_sparse(path, mat, cells=None, genes=None, first_row=False):
    if cells is None and genes is None:
        cells, genes = mat.shape
    if isinstance(cells, int):
        cells = np.arange(cells)
    if isinstance(genes, int):
        genes = np.arange(genes)
    with open(path, 'w') as f:
        rows, cols = mat.nonzero()
        if first_row:
            f.write('{} {}\n'.format(len(cells) + len(genes), rows.size))
        for i, j in zip(rows, cols):
            f.write('c{} g{} {:g}\n'.format(cells[i], genes[j], mat[i, j]))


if __name__ == '__main__':
    import pickle

    # dataset = scvi.dataset.CortexDataset('../data/cortex')
    # export_sparse('../data/cortex/cortex_c2g.txt', dataset.X, dataset.nb_cells, dataset.nb_genes)
    # d = dict(
    #     gene_names=dataset.gene_names,
    #     labels=dataset.labels,
    #     label_to_cell_type=dataset.cell_types
    # )
    # with open('../data/cortex/anno.pickle', 'wb') as f:
    #     pickle.dump(d, f)

    edge_sampler = VoseAlias(np.arange(5),  np.array([0.1, 0.2, 0.3, 0.25, 0.15]))
    sampled_edges = edge_sampler.sample_n(10000, np.random.default_rng())
    print([(sampled_edges == i).mean() for i in range(5)])
