import threading

import anndata
import numpy as np
import torch
from scipy.sparse import csr_matrix


class CellSampler(threading.Thread):
    def __init__(self, adata: anndata.AnnData, args,
                 n_epochs=np.inf, rng=None):
        super().__init__(daemon=True)
        self.n_cells = adata.n_obs
        self.batch_size = args.batch_size
        self.n_epochs = n_epochs
        self.is_sparse = isinstance(adata.X, csr_matrix)
        self.norm_cells = args.norm_cells
        self.X = adata.X
        self.supervised = args.max_supervised_weight > 0
        self.rng = rng
        if self.supervised:
            cell_types = list(adata.obs.cell_types.unique())
            self.cell_type_indices = adata.obs.cell_types.apply(lambda x: cell_types.index(x))
        if self.is_sparse:
            self.library_size = adata.X.sum(1)
        else:
            self.library_size = adata.X.sum(1, keepdims=True)
        self.sample_batches = args.input_batch_id or args.batch_scaling

        self.pipeline = Pipeline()
        if self.sample_batches:
            self.batch_indices = adata.obs.batch_indices.astype(int).values

    def run(self):
        if self.batch_size <= self.n_cells:
            self._low_batch_size()
        else:
            self._high_batch_size()

    def _high_batch_size(self):
        count = 0
        X = torch.FloatTensor(self.X.todense() if self.is_sparse else self.X)
        library_size = torch.FloatTensor(self.library_size)
        cell_indices = torch.arange(0, self.n_cells, dtype=torch.long)
        result_dict = dict(cells=X, library_size=library_size, cell_indices=cell_indices)
        if self.sample_batches:
            result_dict['batch_indices'] = torch.LongTensor(self.batch_indices)
        if self.supervised:
            result_dict['cell_type_indices'] = torch.LongTensor(self.cell_type_indices)
        while count < self.n_epochs:
            count += 1
            self.pipeline.set_message(result_dict)

    def _low_batch_size(self):
        entry_index = 0
        count = 0
        cell_range = np.arange(self.n_cells)
        np.random.shuffle(cell_range)
        while count < self.n_epochs:
            if self.rng is not None:
                batch = self.rng.choice(cell_range, size=self.batch_size)
            else:
                if entry_index + self.batch_size >= self.n_cells:
                    count += 1
                    batch = cell_range[entry_index:]
                    np.random.shuffle(cell_range)
                    excess = entry_index + self.batch_size - self.n_cells
                    if excess > 0 and count < self.n_epochs:
                        batch = np.append(batch, cell_range[:excess], axis=0)
                    entry_index = excess
                else:
                    batch = cell_range[entry_index: entry_index + self.batch_size]
                    entry_index += self.batch_size
            library_size = torch.FloatTensor(self.library_size[batch])
            X = self.X[batch, :]
            if self.is_sparse:
                X = X.todense()
            cells = torch.FloatTensor(X)
            cell_indices = torch.LongTensor(batch)
            result_dict = dict(cells=cells, library_size=library_size, cell_indices=cell_indices)
            if self.sample_batches:
                result_dict['batch_indices'] = torch.LongTensor(self.batch_indices[batch])
            if self.supervised:
                result_dict['cell_type_indices'] = torch.LongTensor(self.cell_type_indices[batch])
            self.pipeline.set_message(result_dict)


class CellSamplerPool:
    def __init__(self, n_samplers, adata: anndata.AnnData, args, n_epochs=np.inf):
        self.samplers = [
            CellSampler(adata, args,
                        n_epochs=n_epochs,
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


from torch.utils.data import Dataset
class CellDataset(Dataset):
    def __init__(self, adata: anndata.AnnData, args):
        super().__init__()
        self.X = adata.X
        self.is_sparse = isinstance(adata.X, csr_matrix)
        self.norm_cells = args.norm_cells
        if self.is_sparse:
            self.library_size = adata.X.sum(1)
        else:
            self.library_size = adata.X.sum(1, keepdims=True)
        if self.norm_cells:
            self.X = self.X / self.library_size
            if self.is_sparse:
                self.X = csr_matrix(self.X)

    
    def __getitem__(self, index):
        if self.is_sparse:
            return self.X.getrow(index).toarray()[0]
        else:
            return self.X[index]

    def __len__(self):
        return self.X.shape[0]


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


if __name__ == '__main__':
    from time import time
    from tqdm import tqdm
    from arg_parser import args
    adata = anndata.read_h5ad('../data/HumanPancreas/HumanPancreas.h5ad')
    sampler = CellSamplerPool(4, adata, args)
    sampler.start()
    start = time()
    current = 0
    for _ in tqdm(range(400)):
        sampler.pipeline.get_message()

    end = time()
    print(end - start)