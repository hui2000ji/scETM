import threading
from typing import Any, Iterator, List, Mapping, Union

import anndata
import numpy as np
import pandas as pd
import torch
import torch.sparse
from scipy.sparse import spmatrix


class CellSampler():
    """An iterable cell dataset for minibatch sampling.

    Attributes:
        n_cells: number of cells in the dataset.
        batch_size: size of each sampled minibatch.
        n_epochs: number of epochs to sample before raising StopIteration.
        X: a (dense or sparse) matrix containing the cell-gene matrix.
        is_sparse: whether self.X is a sparse matrix.
        shuffle: whether to shuffle the dataset at the beginning of each epoch.
            When n_cells >= batch_size, this attribute is ignored.
        rng: the random number generator. Could be None if shuffle is False.
        library_size: a (dense or sparse) vector storing the library size for
            each cell.
        sample_batch_id: whether to yield batch indices in each sample.
        batch_indices: a (dense or sparse) vector storing the batch indices
            for each cell. Only present if sample_batch_id is True.
    """

    def __init__(self,
        adata: anndata.AnnData,
        batch_size: int,
        sample_batch_id: bool = False,
        n_epochs: Union[float, int] = np.inf,
        rng: Union[None, np.random.Generator] = None,
        batch_col: str = 'batch_indices',
        shuffle: bool = True
    ) -> None:
        """Initializes the CellSampler object.

        Args:
            adata: an AnnData object storing the scRNA-seq dataset.
            batch_size: size of each sampled minibatch.
            sample_batch_id: whether to yield batch indices in each sample.
            n_epochs: number of epochs to sample before raising StopIteration.
            rng: the random number generator.
                Could be None if shuffle is False.
            batch_col: a key in adata.obs to the batch column. Only used when
                sample_batch_id is True.
            shuffle: whether to shuffle the dataset at the beginning of each
                epoch. When n_cells >= batch_size, this attribute is ignored.
        """

        self.n_cells: int = adata.n_obs
        self.batch_size: int = batch_size
        self.n_epochs: Union[int, float] = n_epochs
        self.is_sparse: bool = isinstance(adata.X, spmatrix)
        self.X: Union[np.ndarray, spmatrix] = adata.X
        if shuffle:
            self.rng: Union[None, np.random.Generator] = rng or np.random.default_rng()
        else:
            self.rng: Union[None, np.random.Generator] = None
        self.shuffle: bool = shuffle
        if self.is_sparse:
            self.library_size: Union[spmatrix, np.ndarray] = adata.X.sum(1)
        else:
            self.library_size: Union[spmatrix, np.ndarray] = adata.X.sum(1, keepdims=True)
        self.sample_batch_id: bool = sample_batch_id
        if self.sample_batch_id:
            assert batch_col in adata.obs, f'{batch_col} not in adata.obs'
            self.batch_indices: pd.Series = adata.obs[batch_col].astype('category').cat.codes
            
    def __iter__(self) -> Iterator[Mapping[str, torch.Tensor]]:
        """Creates an iterator.

        If self.n_cells <= self.batch_size, simply returns the whole batch for
        self.n_epochs times.
        Otherwise, randomly or sequentially (depending on self.shuffle) sample
        minibatches of size self.batch_size.

        Yields:
            A dict mapping tensor names to tensors. The returned tensors
            include (B for batch_size, G for #genes):
                * X: the cell-gene matrix of shape [B, G].
                * library_size: total #genes for each cell [B].
                * cell_indices: the cell indices in the original dataset [B].
                * batch_indices (optional): the batch indices of each cell [B].
                    Is only returned if self.sample_batch_id is True.
        """

        if self.batch_size < self.n_cells:
            return self._low_batch_size()
        else:
            return self._high_batch_size()

    def _high_batch_size(self) -> Iterator[Mapping[str, torch.Tensor]]:
        """The iterator for the high batch size case.

        Simply returns the whole batch for self.n_epochs times.
        """

        count = 0
        X = torch.FloatTensor(self.X.todense() if self.is_sparse else self.X)
        library_size = torch.FloatTensor(self.library_size)
        cell_indices = torch.arange(0, self.n_cells, dtype=torch.long)
        result_dict = dict(cells=X, library_size=library_size, cell_indices=cell_indices)
        if self.sample_batch_id:
            result_dict['batch_indices'] = torch.LongTensor(self.batch_indices)
        while count < self.n_epochs:
            count += 1
            yield result_dict

    def _low_batch_size(self) -> Iterator[Mapping[str, torch.Tensor]]:
        """The iterator for the low batch size case.

        Randomly or sequentially (depending on self.shuffle) sample minibatches
        of size self.batch_size.
        """

        entry_index = 0
        count = 0
        cell_range = np.arange(self.n_cells)
        if self.shuffle:
            self.rng.shuffle(cell_range)
        while count < self.n_epochs:
            if entry_index + self.batch_size >= self.n_cells:
                count += 1
                batch = cell_range[entry_index:]
                if self.shuffle:
                    self.rng.shuffle(cell_range)
                excess = entry_index + self.batch_size - self.n_cells
                if excess > 0 and count < self.n_epochs:
                    batch = np.append(batch, cell_range[:excess], axis=0)
                    entry_index = excess
                else:
                    entry_index = 0
            else:
                batch = cell_range[entry_index: entry_index + self.batch_size]
                entry_index += self.batch_size

            library_size = torch.FloatTensor(self.library_size[batch])
            X = self.X[batch, :]
            if self.is_sparse:
                # X = X.tocoo()
                # cells = torch.sparse.FloatTensor(torch.LongTensor([X.row, X.col]), torch.FloatTensor(X.data), X.shape)
                cells = torch.FloatTensor(X.todense())
            else:
                cells = torch.FloatTensor(X)
            cell_indices = torch.LongTensor(batch)
            result_dict = dict(cells=cells, library_size=library_size, cell_indices=cell_indices)
            if self.sample_batch_id:
                result_dict['batch_indices'] = torch.LongTensor(self.batch_indices[batch])
            yield result_dict


class ThreadedCellSampler(threading.Thread):
    """A wrapped cell sampler for multi-threaded minibatch sampling.

    Attributes:
        sampler: the underlying iterable cell dataset.
        pipeline: the pipeline connecting the sampler (producer) and the caller
            (consumer).
    """

    def __init__(self,
        adata: anndata.AnnData,
        batch_size: int,
        sample_batch_id: bool = False,
        n_epochs: Union[float, int] = np.inf,
        rng: Union[None, np.random.Generator] = None,
        batch_col: str = 'batch_indices',
        shuffle: bool = True
    ) -> None:
        """Initializes the ThreadedCellSampler object.

        This will initialize the attributes and then start the thread by
        calling self.start(), which arranges the invoke of self.run().

        Args:
            adata: an AnnData object storing the scRNA-seq dataset.
            batch_size: size of each sampled minibatch.
            sample_batch_id: whether to yield batch indices in each sample.
            n_epochs: number of epochs to sample before raising StopIteration.
            rng: the random number generator.
                Could be None if shuffle is False.
            batch_col: a key in adata.obs to the batch column.
            shuffle: whether to shuffle the dataset at the beginning of each
                epoch. When n_cells >= batch_size, this attribute is ignored.
        """

        super().__init__(daemon=True)
        self.sampler: CellSampler = CellSampler(adata, batch_size, sample_batch_id, n_epochs, rng, batch_col, shuffle)
        self.pipeline: Pipeline = Pipeline()
        self.start()

    def run(self) -> None:
        """Iterates over self.sampler and write the sampled data to self.pipeline.
        When there are no more samples, write None to self.pipeline.
        """

        for result_dict in self.sampler:
            self.pipeline.set_message(result_dict)
        self.pipeline.set_message(None)

    def __iter__(self) -> Iterator[Mapping[str, torch.Tensor]]:
        """Creates an iterator.

        If n_cells <= batch_size, simply returns the whole batch for n_epochs
        times. Otherwise, randomly or sequentially (depending on shuffle)
        sample minibatches of size batch_size.

        Yields:
            A dict mapping tensor names to tensors. The returned tensors
            include (B for batch_size, G for #genes):
                * X: the cell-gene matrix of shape [B, G].
                * library_size: total #genes for each cell [B].
                * cell_indices: the cell indices in the original dataset [B].
                * batch_indices (optional): the batch indices of each cell [B].
                    Is only returned if self.sample_batch_id is True.
        """

        return self.iterator()

    def iterator(self) -> Iterator[Mapping[str, torch.Tensor]]:
        """Iteratively get sampled batches from self.pipeline.
        Upon receiving None, meaning there are no more samples, stop iteration.
        """

        while True:
            try:
                result_dict = self.pipeline.get_message()
            except Exception:
                break
            if result_dict is None:
                break
            yield result_dict


class MultithreadedCellSampler:
    """A multithreaded iterable cell dataset for minibatch sampling.

    This class should be only used in training and when the #cells in the
    dataset is larger than the batch size. In other cases, using CellSampler
    would suffice.
    Note that the shuffle parameter is fixed to True.

    Attributes
        n_samplers: #ThreadedCellSamplers (#threads) in the pool.
        samplers: a list of ThreadedCellSamplers.
        iterators: a list of iterators derived from samplers.
        current_iterator: index of the current iterator. Could take values
            from 0 to n_samplers - 1.
    """

    def __init__(self,
        adata: anndata.AnnData,
        batch_size: int,
        n_samplers: int = 4,
        sample_batch_id: bool = False,
        n_epochs: Union[float, int] = np.inf,
        batch_col: str = 'batch_indices'
    ) -> None:
        """Initializes the MultithreadedCellSampler object.

        Args:
            adata: an AnnData object storing the scRNA-seq dataset.
            batch_size: size of each sampled minibatch.
            n_samplers: #ThreadedCellSamplers (#threads) in the pool.
            sample_batch_id: whether to yield batch indices in each sample.
            n_epochs: number of epochs to sample before raising StopIteration.
            rng: the random number generator.
                Could be None if shuffle is False.
            batch_col: a key in adata.obs to the batch column.
        """

        self.n_samplers: int = n_samplers
        self.samplers: List[ThreadedCellSampler] = [ThreadedCellSampler(
            adata,
            batch_size,
            sample_batch_id,
            n_epochs = n_epochs / self.n_samplers,
            rng = np.random.default_rng(i * 100 + np.random.randint(100)),
            batch_col=batch_col,
            shuffle=True
        ) for i in range(self.n_samplers)]
        self.iterators: List[Iterator[Mapping[str, torch.Tensor]]] = [iter(sampler) for sampler in self.samplers]
        self.current_iterator: int = 0

    def __iter__(self) -> Iterator[Mapping[str, torch.Tensor]]:
        """Creates an iterator.

        The iterator yields the results from the samplers in a round-robin
        fashion. Each sampler randomly samples minibatches of size batch_size.

        Yields:
            A dict mapping tensor names to tensors. The returned tensors
            include (B for batch_size, G for #genes):
                * X: the cell-gene matrix of shape [B, G].
                * library_size: total #genes for each cell [B].
                * cell_indices: the cell indices in the original dataset [B].
                * batch_indices (optional): the batch indices of each cell [B].
                    Is only returned if self.sample_batch_id is True.
        """

        return self.iterator()

    def iterator(self) -> Iterator[Mapping[str, torch.Tensor]]:
        """Iteratively get sampled batches from one of the samplers.
        Upon receiving StopIteration, meaning there are no more samples,
        stop iteration.
        """

        while True:
            self.current_iterator += 1
            if self.current_iterator == self.n_samplers:
                self.current_iterator = 0
            try:
                result_dict = next(self.iterators[self.current_iterator])
            except StopIteration:
                break
            yield result_dict

    def join(self, seconds: Union[float, None] = None) -> None:
        """Joins all sampler threads.

        Args:
            seconds: seconds to wait before thread termination. If None, block
                until termination.
        """

        for sampler in self.samplers:
            sampler.join(seconds)


class Pipeline:
    """
    A single element pipeline between the producer and the consumer.

    Attributes:
        message: message sent from the producer to the consumer.
        producer_lock: a lock for the producer. Locked when the producer begins
            writing. Released when the consumer finished reading.
        consumer_lock: a lock for the consumer. Locked at the beginning or when
            the consumer begins reading. Released when the producer finished
            writing.
    """

    def __init__(self) -> None:
        """Initializes the Pipeline object.

        Initialize the attributes, then lock the consumer_lock.
        """

        self.message: Any = None
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()

    def get_message(self) -> Any:
        """Reads message from the producer. Called by the consumer.
        """

        self.consumer_lock.acquire()
        message = self.message
        self.producer_lock.release()
        return message

    def set_message(self, message: Any) -> None:
        """Writes message to the consumer. Called by the producer.
        """

        self.producer_lock.acquire()
        self.message = message
        self.consumer_lock.release()
