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
        self.table_prob = np.zeros_like(self.weights, dtype=np.float64)  # probability table
        self.list_prob = None
        self.table_alias = np.zeros_like(self.weights, dtype=np.int64)  # alias table
        self.alias_init()

    def alias_init(self):
        """
        Construct probability and alias tables for the distribution.
        """
        # Initialise variables
        n = len(self.weights)

        # Construct and sort the scaled probabilities into their appropriate stacks
        print("1/2. Building and sorting scaled probabilities for alias table...")
        scaled_prob = (self.weights * n).astype(np.float64)  # scaled probabilities
        small_mask = scaled_prob < 1.
        small = small_mask.nonzero()[0].tolist()  # stack for probabilities smaller that 1
        large = (~small_mask).nonzero()[0].tolist()  # stack for probabilities greater than or equal to 1

        print("2/2. Building alias table...")
        # Construct the probability and alias tables
        while small and large:
            s = small.pop()
            l = large.pop()

            self.table_prob[s] = scaled_prob[s]
            self.table_alias[s] = l

            scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - 1.  # Decimal(1)

            if scaled_prob[l] < 1.:
                small.append(l)
            else:
                large.append(l)

        # The remaining outcomes (of one stack) must have probability 1
        while large:
            self.table_prob[large.pop()] = 1.  # Decimal(1)

        while small:
            self.table_prob[small.pop()] = 1.  # Decimal(1)

        self.list_prob = np.arange(n, dtype=np.int64)

    def alias_generation(self):
        """
        Yields a random outcome from the distribution.
        """
        # Determine which column of table_prob to inspect
        col = random.choice(self.list_prob)
        # Determine which outcome to pick in that column
        if self.table_prob[col] >= random.uniform(0, 1):
            return col
        else:
            return self.table_alias[col]

    def sample_n(self, size, rng):
        """
        Yields a sample of size n from the distribution, and print the results to stdout.
        """
        col = rng.choice(self.list_prob, size=size)
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

    dataset = scvi.dataset.CortexDataset('../data/cortex')
    export_sparse('../data/cortex/cortex_c2g.txt', dataset.X, dataset.nb_cells, dataset.nb_genes)
    d = dict(
        gene_names=dataset.gene_names,
        labels=dataset.labels,
        label_to_cell_type=dataset.cell_types
    )
    with open('../data/cortex/anno.pickle', 'wb') as f:
        pickle.dump(d, f)

    # edges, edge_weights, gene_degrees, n_cells, n_genes, X = make_distribution('../data/cortex/cortex_c2g.txt', 0.75)
    # edge_sampler = VoseAlias(edges, edge_weights)
    # node_sampler = VoseAlias(np.arange(n_genes, dtype=np.int32), gene_degrees)
    # sampled_edges = edge_sampler.sample_n(10)
    # cells, genes, neg_genes = sample_batch(sampled_edges, 5, node_sampler)
    # print(cells, genes, neg_genes, sep='\n\n')
