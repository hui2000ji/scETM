import threading

import anndata
import numpy as np
import torch

from data_utils import VoseAlias


class CellSampler(threading.Thread):
    def __init__(self, n_cells, batch_size, device, n_epochs=np.inf):
        super().__init__(daemon=True)
        self.n_cells = n_cells
        self.batch_size = batch_size
        self.device = device
        self.n_epochs = n_epochs
        self.pipeline = Pipeline()

    def run(self):
        if self.batch_size < self.n_cells:
            self._low_batch_size()
        else:
            self._high_batch_size()

    def _high_batch_size(self):
        count = 0
        cells = np.arange(self.n_cells)
        while count < self.n_epochs:
            count += 1
            batch = torch.LongTensor(cells).to(self.device)
            self.pipeline.set_message(batch)

    def _low_batch_size(self):
        entry_index = 0
        count = 0
        cells = np.arange(self.n_cells)
        np.random.shuffle(cells)
        while count < self.n_epochs:
            count += 1
            if entry_index + self.batch_size >= self.n_cells:
                batch = cells[entry_index:]
                np.random.shuffle(cells)
                excess = entry_index + self.batch_size - self.n_cells
                if excess > 0:
                    batch = np.append(batch, cells[:excess], axis=0)
                entry_index = excess
            else:
                batch = cells[entry_index: entry_index + self.batch_size]
                entry_index += self.batch_size
            batch = torch.LongTensor(batch).to(self.device)
            self.pipeline.set_message(batch)


class EdgeSampler(threading.Thread):
    def __init__(self, edges, batch_size, row_size, device, n_epochs=np.inf, n_neg_samples=0, gene_prob=None):
        super().__init__(daemon=True)
        self.edge_probs = edges / edges.sum()
        self.edges_range = np.arange(edges.size)
        self.batch_size = batch_size
        self.n_neg_samples = n_neg_samples
        self.row_size = row_size
        self.device = device
        self.n_epochs = n_epochs
        self.pipeline = Pipeline()

        if self.n_neg_samples:
            assert gene_prob is not None
            self.gene_prob = gene_prob

    def run(self):
        count = 0
        while count < self.n_epochs:
            count += 1
            shuffled_edges = np.random.choice(
                self.edges_range, size=self.batch_size, p=self.edge_probs)
            batch = np.array([(choice // self.row_size, choice % self.row_size)
                              for choice in shuffled_edges], dtype=np.int32)
            cells, genes = batch[:, 0], batch[:, 1]
            if self.n_neg_samples:
                neg_genes = np.random.choice(np.arange(self.row_size), size=(
                    self.batch_size, self.n_neg_samples), p=self.gene_prob)
                same_genes = genes[:, np.newaxis] == neg_genes
                num_same_genes = same_genes.sum()
                while num_same_genes > 0:
                    neg_genes[same_genes] = np.random.choice(
                        np.arange(self.row_size), size=(num_same_genes,), p=self.gene_prob)
                    same_genes = genes[:, np.newaxis] == neg_genes
                    num_same_genes = same_genes.sum()
                neg_genes = torch.LongTensor(neg_genes).to(self.device)
                cells = torch.LongTensor(cells).to(self.device)
                genes = torch.LongTensor(genes).to(self.device)
                self.pipeline.set_message((cells, genes, neg_genes))
            else:
                cells = torch.LongTensor(cells).to(self.device)
                genes = torch.LongTensor(genes).to(self.device)
                self.pipeline.set_message((cells, genes))


class NonZeroEdgeSampler(threading.Thread):
    def __init__(self, adata: anndata.AnnData, args, device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu"),
                 n_epochs=np.inf):
        super().__init__(daemon=True)
        self.rows, self.cols = adata.X.nonzero()
        self.weights = adata.X[self.rows, self.cols]
        self.weights = self.weights / self.weights.sum()
        self.edges_range = np.arange(self.rows.size)
        self.row_size = adata.n_vars
        self.genes_range = np.arange(self.row_size)
        self.batch_size = args.batch_size
        self.n_neg_samples = args.neg_samples
        self.device = device
        self.n_epochs = n_epochs
        self.pipeline = Pipeline()
        self.max_lambda = args.max_lambda
        if self.max_lambda:
            self.batch_indices = adata.obs.batch_indices.values

        if self.n_neg_samples:
            self.gene_prob = adata.X.sum(0) ** args.neg_power
            self.gene_prob = self.gene_prob / self.gene_prob.sum()

    def run(self):
        count = 0
        while count < self.n_epochs:
            count += 1
            mask = np.random.choice(
                self.edges_range, size=self.batch_size, p=self.weights)
            cells, genes = self.rows[mask], self.cols[mask]
            if self.n_neg_samples:
                neg_genes = np.random.choice(self.genes_range, size=(
                    self.batch_size, self.n_neg_samples), p=self.gene_prob)
                same_genes = genes[:, np.newaxis] == neg_genes
                num_same_genes = same_genes.sum()
                while num_same_genes > 0:
                    neg_genes[same_genes] = np.random.choice(
                        self.genes_range, size=(num_same_genes,), p=self.gene_prob)
                    same_genes = genes[:, np.newaxis] == neg_genes
                    num_same_genes = same_genes.sum()
                neg_genes_tensor = torch.LongTensor(neg_genes).to(self.device)
                cells_tensor = torch.LongTensor(cells).to(self.device)
                genes_tensor = torch.LongTensor(genes).to(self.device)
                result_dict = dict(
                    cells=cells_tensor, genes=genes_tensor, neg_genes=neg_genes_tensor)
            else:
                cells_tensor = torch.LongTensor(cells).to(self.device)
                genes_tensor = torch.LongTensor(genes).to(self.device)
                result_dict = dict(cells=cells_tensor, genes=genes_tensor)
            if self.max_lambda:
                result_dict['batch_indices'] = self.batch_indices[cells]
            self.pipeline.set_message(result_dict)


class VAEdgeSampler(threading.Thread):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu"), n_epochs=np.inf,
                 edge_sampler=None, node_sampler=None, rng=None):
        super().__init__(daemon=True)
        if edge_sampler is None or node_sampler is None or rng is None:
            rows, cols = adata.X.nonzero()
            weights = adata.X[rows, cols]
            weights = weights / weights.sum()
            gene_prob = adata.X.sum(0) ** args.neg_power
            gene_prob = gene_prob / gene_prob.sum()
            self.edge_sampler = VoseAlias(np.vstack((rows, cols)).T, weights)
            self.node_sampler = VoseAlias(
                np.arange(adata.n_vars, dtype=np.int32), gene_prob)
            self.rng = np.random.default_rng()
        elif edge_sampler is not None and node_sampler is not None and rng is not None:
            self.edge_sampler = edge_sampler
            self.node_sampler = node_sampler
            self.rng = rng
        else:
            raise ValueError(
                "Either provide both edge_sampler and node_sampler, or provide neither")
        self.batch_size = args.batch_size
        self.n_neg_samples = args.neg_samples
        self.device = device
        self.n_epochs = n_epochs
        self.pipeline = Pipeline()
        self.max_lambda = args.max_lambda
        if self.max_lambda:
            self.batch_indices = adata.obs.batch_indices.values

    def run(self):
        count = 0
        while count < self.n_epochs:
            count += 1
            sampled_edges = self.edge_sampler.sample_n(
                self.batch_size, self.rng)
            cells, genes = sampled_edges[:, 0], sampled_edges[:, 1]
            expanded_genes = genes[:, np.newaxis]
            batch_size = cells.shape[0]
            neg_genes = self.node_sampler.sample_n(self.batch_size * self.n_neg_samples, self.rng) \
                .reshape(self.batch_size, self.n_neg_samples)
            same_genes = expanded_genes == neg_genes
            num_same_genes = same_genes.sum()
            while num_same_genes > 0:
                neg_genes[same_genes] = self.node_sampler.sample_n(
                    num_same_genes, self.rng)
                same_genes = expanded_genes == neg_genes
                num_same_genes = same_genes.sum()
            cells_tensor = torch.LongTensor(cells).to(self.device)
            genes_tensor = torch.LongTensor(genes).to(self.device)
            neg_genes_tensor = torch.LongTensor(neg_genes).to(self.device)
            result_dict = dict(cells=cells_tensor,
                               genes=genes_tensor, neg_genes=neg_genes_tensor)
            if self.max_lambda:
                result_dict['batch_indices'] = self.batch_indices[cells]
            self.pipeline.set_message(result_dict)


class Pipeline:
    """
    Class to allow a single element pipeline between producer and consumer.
    """

    def __init__(self):
        self.message = 0
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()

    def get_message(self):
        self.consumer_lock.acquire()
        message = self.message
        self.producer_lock.release()
        return message

    def set_message(self, message):
        self.producer_lock.acquire()
        self.message = message
        self.consumer_lock.release()


class VAEdgeSamplerPool:
    def __init__(self, n_samplers, adata: anndata.AnnData, args,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu"), n_epochs=np.inf):
        rows, cols = adata.X.nonzero()
        weights = adata.X[rows, cols]
        weights = weights / weights.sum()
        gene_prob = adata.X.sum(0) ** args.neg_power
        gene_prob = gene_prob / gene_prob.sum()
        edge_sampler = VoseAlias(np.vstack((rows, cols)).T, weights)
        node_sampler = VoseAlias(
            np.arange(adata.n_vars, dtype=np.int32), gene_prob)
        self.samplers = [
            VAEdgeSampler(adata, args,
                          device=device,
                          n_epochs=n_epochs,
                          edge_sampler=edge_sampler,
                          node_sampler=node_sampler,
                          rng=np.random.default_rng(i * 100 + np.random.randint(100))
                         ) for i in range(n_samplers)]
        self.current = 0
        self.n_samplers = n_samplers

    def start(self):
        for sampler in self.samplers:
            sampler.start()

    @property
    def pipeline(self):
        pl = self.samplers[self.current].pipeline
        self.current += 1
        if self.current == self.n_samplers:
            self.current = 0
        return pl

    def join(self, seconds):
        for sampler in self.samplers:
            sampler.join(seconds)


if __name__ == '__main__':
    adata = anndata.AnnData(X=np.arange(6).reshape(2, 3))
    from my_parser import parser
    args = parser.parse_args()

    cell_gene_sampler = VAEdgeSampler(adata, args)
    cell_gene_sampler.start()

    edge_bins = np.zeros((2, 3), dtype=np.int64)
    gene_bins = np.zeros(3, dtype=np.int64)
    print('started edge sampler')
    for _ in range(100):
        data_dict = cell_gene_sampler.pipeline.get_message()
        cells, genes, neg_genes = data_dict['cells'], data_dict['genes'], data_dict['neg_genes']
        for j in range(3):
            for i in range(2):
                edge_bins[i, j] += ((cells == i) & (genes == j)).sum()
            gene_bins[j] += (neg_genes == j).sum()
    print(edge_bins)
    print()
    print(gene_bins)
