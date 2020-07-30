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


def read_LINE(path, n_cells, n_genes):
    with open(path, 'r') as f:
        n_nodes, emb_dim = map(int, f.readline().split())
        cell_emb = np.zeros((n_cells, emb_dim), dtype=np.float32)
        gene_emb = np.zeros((n_genes, emb_dim), dtype=np.float32)
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            node_name, emb = tokens[0], list(map(float, tokens[1:]))
            if node_name[0] == 'c':
                idx = int(node_name[1:])
                cell_emb[idx, :] = emb
            elif node_name[0] == 'g':
                idx = int(node_name[1:])
                gene_emb[idx, :] = emb
            else:
                raise RuntimeError("Invalid node name")
        return cell_emb, gene_emb


class VoseAlias(object):
    """
    Adding a few modifs to https://github.com/asmith26/Vose-Alias-Method
    """

    def __init__(self, keys, weights):
        """
        (VoseAlias, dict) -> NoneType
        """
        self.keys = keys
        self.weights = weights
        self.n = len(self.weights)
        self.table_prob = (self.weights * self.n).astype(np.float64)  # scaled probabilities
        self.list_prob = None
        self.table_alias = np.zeros_like(self.weights, dtype=np.int64)  # alias table
        self.alias_init()

    def alias_init(self):
        """
        Construct probability and alias tables for the distribution.
        """
        # Construct and sort the scaled probabilities into their appropriate stacks
        index = np.arange(self.n)
        small_mask = self.table_prob < 1.
        small_index = index[small_mask]
        large_index = index[~small_mask]
        while small_index.size and large_index.size:
            count = min(small_index.size, large_index.size)
            small, small_index = np.split(small_index, (count,))
            large, large_index = np.split(large_index, (count,))

            self.table_alias[small] = large
            self.table_prob[large] += self.table_prob[small] - 1

            small_mask = self.table_prob[large] < 1.
            small_index = np.concatenate((small_index, large[small_mask]))
            large_index = np.concatenate((large_index, large[~small_mask]))
        
        self.table_alias[small_index] = small_index
        self.table_alias[large_index] = large_index

    def alias_generation(self):
        """
        Yields a random outcome from the distribution.
        """
        # Determine which column of table_prob to inspect
        col = random.randint(self.n)
        # Determine which outcome to pick in that column
        if self.table_prob[col] >= random.uniform(0, 1):
            return col
        else:
            return self.table_alias[col]

    def sample_n(self, size, rng):
        """
        Yields a sample of size n from the distribution, and print the results to stdout.
        """
        col = rng.integers(0, self.n, size=size)
        ref = rng.uniform(0., 1., size=size)
        mask_less_than_ref = self.table_prob[col] < ref
        col[mask_less_than_ref] = self.table_alias[col][mask_less_than_ref]
        return self.keys[col]


def make_distribution(graph_path, power=0.75, generate_dense=False):
    # edge_dist_dict = collections.defaultdict(float)
    # node_dist_dict = collections.defaultdict(float)

    weight_sum = 0
    neg_prob_sum = 0

    n_lines = 0

    with open(graph_path, "r") as graph_file:
        n_cells, n_genes = map(int, graph_file.readline().split())
        for l in graph_file:
            n_lines += 1

    print("Reading edge list file...")
    X = np.zeros((n_cells, n_genes), dtype=np.float32) if generate_dense else None
    edges = np.zeros((n_lines, 2), dtype=np.int32)
    edge_weights = np.zeros(n_lines, dtype=np.float32)
    gene_degrees = np.zeros(n_genes, dtype=np.float32)
    with open(graph_path, "r") as graph_file:
        graph_file.readline()
        for i, l in enumerate(tqdm(graph_file, total=n_lines, ncols=80)):
            line = l.strip().split()
            node1, node2, weight = int(line[0][1:]), int(line[1][1:]), float(line[2])

            # edge_dist_dict[(node1, node2)] = weight
            # weight_sum += weight

            edges[i] = (node1, node2)
            edge_weights[i] = weight

            gene_degrees[node2] = weight ** power

            # powered_weight = weight ** power
            # node_dist_dict[node2] += powered_weight
            # neg_prob_sum += powered_weight

            if generate_dense:
                X[node1, node2] = weight

    # for node, powered_out_degree in node_dist_dict.items():
    #     node_dist_dict[node] = powered_out_degree / neg_prob_sum

    # for edge, weight in edge_dist_dict.items():
    #     edge_dist_dict[edge] = weight / weight_sum

    # return edge_dist_dict, node_dist_dict, n_cells, n_genes, X

    edge_weights = edge_weights / edge_weights.sum()
    gene_degrees = gene_degrees / gene_degrees.sum()
    # return {(row, col): weight for row, col, weight in zip(rows, cols, edge_weights)}, {i: deg for i, deg in enumerate(gene_degrees)}, n_cells, n_genes, X
    return edges, edge_weights, gene_degrees, n_cells, n_genes, X


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
