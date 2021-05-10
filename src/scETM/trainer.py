import os
import copy
from pathlib import Path
import pandas as pd
import time
import random
from typing import DefaultDict, IO, List, Sequence, Union, Tuple
import psutil
import logging
from collections import defaultdict

import numpy as np
import anndata
import torch
from torch import nn
from torch import optim

from scETM.batch_sampler import CellSampler, MultithreadedCellSampler
from scETM.eval_utils import evaluate
from scETM.model import BaseCellModel, scETM
from scETM.logging_utils import initialize_logger, log_arguments

_logger = logging.getLogger(__name__)


class UnsupervisedTrainer:
    """Unsupervised trainer for single-cell modeling.

    Sets up the random seed, dataset split, optimizer and logger, and executes
    training and evaluation loop.

    Attributes:
        model: the model to be trained.
        adata: the intact single-cell dataset.
        train_adata: the training data. Contains (1 - test_ratio) × 100% of
            adata.
        test_adata: the test data. Contains test_ratio × 100% of adata.
        optimizer: the optimizer used to train the model.
        lr: the current learning rate.
        init_lr: the initial learning rate.
        lr_decay: the negative log of the decay rate of the learning rate.
            After each training step, lr = lr * exp(-lr_decay).
        batch_size: the training batch size.
        steps_per_epoch: #training steps to cover an epoch.
        device: device the model is on.
        step: current step.
        epoch: current epoch.
        seed: random seed.
        train_instance_name: name for this train instance for checkpointing.
        ckpt_dir: directory to store the logs, the checkpoints and the plots.
    """

    @log_arguments
    def __init__(self,
        model: BaseCellModel,
        adata: anndata.AnnData,
        ckpt_dir: Union[str, None] = None,
        test_ratio: float = 0.,
        data_split_seed: int = 1,
        init_lr: float = 5e-3,
        lr_decay: float = 6e-5,
        batch_size: int = 2000,
        train_instance_name: str = "scETM",
        restore_epoch: int = 0,
        seed: int = -1,
    ) -> None:
        """Initializes the UnsupervisedTrainer object.

        Args:
            model: the model to be trained.
            adata: the intact single-cell dataset.
            ckpt_dir: directory to store the logs, the checkpoints and the
                plots. If training from scratch (restore_epoch = 0), this would
                be the parent directory of the actual directory storing the
                checkpoints (self.ckpt_dir = ckpt_dir / train_instance_name);
                if restoring from checkpoints, this would be the directory
                holding the checkpoint files.
            test_ratio: ratio of the test data in adata.
            init_lr: the initial learning rate.
            lr_decay: the negative log of the decay rate of the learning rate.
                After each training step, lr = lr * exp(-lr_decay).
            batch_size: the training batch size.
            train_instance_name: name for this train instance for checkpointing.
            restore_epoch: the epoch to restore from ckpt_dir.
            seed: random seed.
        """

        if seed >= 0:
            set_seed(seed)

        self.model: BaseCellModel = model

        self.train_adata = self.test_adata = self.adata = adata
        if test_ratio > 0:
            self.train_adata, self.test_adata = train_test_split(adata, test_ratio, seed=data_split_seed)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=init_lr)
        self.lr = self.init_lr = init_lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.steps_per_epoch = max(self.train_adata.n_obs / self.batch_size, 1)
        self.device = model.device
        self.step = self.epoch = 0
        self.seed = seed

        self.train_instance_name = train_instance_name
        if restore_epoch > 0:
            self.ckpt_dir = ckpt_dir
            self.load_ckpt(restore_epoch, self.ckpt_dir)
        elif ckpt_dir is not None:
            self.ckpt_dir = os.path.join(ckpt_dir, f"{self.train_instance_name}_{time.strftime('%m_%d-%H_%M_%S')}")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            initialize_logger(self.ckpt_dir)
            _logger.info(f'ckpt_dir: {self.ckpt_dir}')
        else:
            self.ckpt_dir = None

    @log_arguments
    def load_ckpt(self, restore_epoch: int, ckpt_dir: Union[str, None] = None) -> None:
        """Loads model checkpoints.

        After loading, self.step, self.epoch and self.lr are set to
        the corresponding values, and the loger will be re-initialized.

        Args:
            restore_epoch: the epoch to restore.
            ckpt_dir: the directory containing the model checkpoints. If None,
                set to self.ckpt_dir.
        """

        if ckpt_dir is None:
            ckpt_dir = self.ckpt_dir
        assert ckpt_dir is not None and os.path.exists(ckpt_dir), f"ckpt_dir {ckpt_dir} does not exist."
        model_ckpt_path = os.path.join(ckpt_dir, f'model-{restore_epoch}')
        self.model.load_state_dict(torch.load(model_ckpt_path))
        _logger.info(f'Parameters restored from {model_ckpt_path}.')
        optim_ckpt_path = os.path.join(ckpt_dir, f'opt-{restore_epoch}')
        self.optimizer.load_state_dict(torch.load(optim_ckpt_path))
        _logger.info(f'Optimizer restored from {optim_ckpt_path}.')
        initialize_logger(self.ckpt_dir)
        _logger.info(f'ckpt_dir: {self.ckpt_dir}')
        self.update_step(restore_epoch * self.steps_per_epoch)

    @staticmethod
    def _get_kl_weight(
        epoch: int,
        n_epochs: int,
        kl_warmup_ratio: float = 1/3,
        min_kl_weight: float = 0.,
        max_kl_weight: float = 1e-7
    ) -> float:
        """Calculates weight of the KL term.

        Args:
            epoch: current epoch.
            n_epochs: the total number of epochs to train the model.
            kl_warmup_ratio: ratio of KL warmup epochs and n_epochs.
            min_kl_weight: minimum weight of the KL term.
            max_kl_weight: maximum weight of the KL term.

        Returns:
            The current weight of the KL term.
        """

        if kl_warmup_ratio:
            return max(min(1., epoch / (n_epochs * kl_warmup_ratio)) * max_kl_weight, min_kl_weight)
        else:
            return max_kl_weight

    def update_step(self, jump_to_step: Union[None, int] = None) -> None:
        """Aligns the current step, epoch and lr to the given step number.

        Args:
            jump_to_step: the step number to jump to. If None, increment the
                step number by one.
        """

        if jump_to_step is None:
            self.step += 1
        else:
            self.step = jump_to_step
        self.epoch = self.step / self.steps_per_epoch
        if self.lr_decay:
            if jump_to_step is None:
                self.lr *= np.exp(-self.lr_decay)
            else:
                self.lr = self.init_lr * np.exp(-jump_to_step * self.lr_decay)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    @log_arguments
    def train(self,
        n_epochs: int = 800,
        eval_every: int = 200,
        n_samplers: int = 4,
        kl_warmup_ratio: float = 1/3,
        min_kl_weight: float = 0.,
        max_kl_weight: float = 1e-7,
        eval: bool = True,
        batch_col: str = "batch_indices",
        save_model_ckpt: bool = True,
        record_log_path: Union[str, None] = None,
        eval_result_log_path: Union[str, None] = None,
        eval_kwargs: Union[None, dict] = None
    ) -> None:
        """Trains the model, optionally evaluates performance and logs results.

        Args:
            n_epochs: the total number of epochs to train the model.
            eval_every: evaluate the model every this many epochs.
            n_samplers: #samplers (#threads) to use to sample training
                minibatches.
            kl_warmup_ratio: ratio of KL warmup epochs and n_epochs.
            min_kl_weight: minimum weight of the KL term.
            max_kl_weight: maximum weight of the KL term.
            eval: whether to evaluate the model.
            batch_col: a key in adata.obs to the batch column.
            save_model_ckpt: whether to save the model checkpoints.
            record_log_path: path to log the training records. If None, do not
                log.
            eval_result_log_path: path to log the evaluation results. If None,
                do not log.
            eval_kwargs: dict to pass to the evaluate function as kwargs.
        """

        default_eval_kwargs = dict(
            batch_col = batch_col,
            plot_fname = f'{self.train_instance_name}_{self.model.clustering_input}',
            plot_dir = self.ckpt_dir
        )
        if eval_kwargs is not None:
            default_eval_kwargs.update(eval_kwargs)
        eval_kwargs = default_eval_kwargs
        
        # set up sampler and dataloader
        if n_samplers == 1 or self.batch_size >= self.train_adata.n_obs:
            sampler = CellSampler(self.train_adata, self.batch_size, sample_batch_id = self.model.need_batch, n_epochs = n_epochs - self.epoch, batch_col = batch_col)
        else:
            sampler = MultithreadedCellSampler(self.train_adata, self.batch_size, n_samplers = n_samplers, sample_batch_id = self.model.need_batch, n_epochs = n_epochs - self.epoch, batch_col = batch_col)
        dataloader = iter(sampler)
        
        # set up the stats recorder
        recorder = _stats_recorder(record_log_path=record_log_path)
        next_ckpt_epoch = int(np.ceil(self.epoch / eval_every) * eval_every)

        while self.epoch < n_epochs:
            # construct hyper_param_dict
            hyper_param_dict = {
                'beta': self._get_kl_weight(self.epoch, n_epochs, kl_warmup_ratio, min_kl_weight, max_kl_weight)
            }

            # construct data_dict
            data_dict = {k: v.to(self.device) for k, v in next(dataloader).items()}

            # train for one step, record tracked items (e.g. loss)
            new_record = self.model.train_step(self.optimizer, data_dict, hyper_param_dict)
            recorder.update(new_record, self.epoch, n_epochs, next_ckpt_epoch)
            self.update_step()

            # log and evaluate
            if self.epoch >= next_ckpt_epoch or self.epoch >= n_epochs:
                _logger.info('=' * 10 + f'Epoch {self.epoch:.0f}' + '=' * 10)

                # log memory cost
                _logger.info(repr(psutil.Process().memory_info()))
                # log current lr and kl_weight
                if self.lr_decay:
                    _logger.info(f'{"lr":12s}: {self.lr}')
                _logger.info(f'{"kl_weight":12s}: {self._get_kl_weight(self.epoch, n_epochs):12.4f}')

                # log statistics of tracked items
                recorder.log_and_clear_record()
                
                if eval:
                    current_eval_kwargs = eval_kwargs.copy()
                    current_eval_kwargs['plot_fname'] = current_eval_kwargs['plot_fname'] + f'_epoch{int(self.epoch)}'
                    if self.test_adata is not self.adata:
                        test_nll = self.model.get_cell_embeddings_and_nll(self.test_adata, self.batch_size, batch_col=batch_col, emb_names=[])
                        if test_nll is not None:
                            _logger.info(f'test nll: {test_nll:7.4f}')
                    else:
                        test_nll = None
                    self.model.get_cell_embeddings_and_nll(self.adata, self.batch_size, batch_col=batch_col, emb_names=[self.model.clustering_input])
                    result = evaluate(adata = self.adata, embedding_key = self.model.clustering_input, **current_eval_kwargs)
                    if eval_result_log_path is not None:
                        with open(eval_result_log_path, 'a+') as f:
                            # ckpt_dir, epoch, test_nll, ari, nmi, k_bet, ebm, time, seed
                            f.write(f'{Path(self.ckpt_dir).name}\t'
                                    f'{self.epoch}\t'
                                    f'{test_nll}\t'
                                    f'{result["ari"]}\t'
                                    f'{result["nmi"]}\t'
                                    f'{result["k_bet"]}\t'
                                    f'{result["ebm"]}\t'
                                    f'{time.strftime("%m_%d-%H_%M_%S")}\t'
                                    f'{self.seed}\n')

                if next_ckpt_epoch and save_model_ckpt and self.ckpt_dir is not None:
                    # checkpointing
                    torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, f'model-{next_ckpt_epoch}'))
                    torch.save(self.optimizer.state_dict(), os.path.join(self.ckpt_dir, f'opt-{next_ckpt_epoch}'))

                _logger.info('=' * 10 + f'End of evaluation' + '=' * 10)
                next_ckpt_epoch += eval_every

        del recorder
        _logger.info("Optimization Finished: %s" % self.ckpt_dir)
        if isinstance(sampler, MultithreadedCellSampler):
            sampler.join(0.1)


class _stats_recorder:
    """A utility class for recording training statistics.

    TODO: integrate with tensorboard.

    Attributes:
        record: the training statistics record.
        fmt: print format for the training statistics.
        log_file: the file stream to write logs to.
    """

    def __init__(self, record_log_path: Union[str, None] = None, fmt: str = "12.4f") -> None:
        """Initializes the statistics recorder.
        
        Args:
            record_log_path: the file path to write logs to.
            fmt: print format for the training statistics.
        """

        self.record: DefaultDict[List] = defaultdict(list)
        self.fmt: str = fmt
        if record_log_path is not None:
            self.log_file: Union[None, IO] = open(record_log_path, 'w')
            self._header_logged: bool = False
        else:
            self.log_file = None

    def update(self, new_record: dict, epoch: float, total_epochs: int, next_ckpt_epoch: int) -> None:
        """Updates the record and prints a \\r-terminated line to the console.

        If self.log_file is not None, this function will also write a line of
        log to self.log_file.

        Args:
            new_record: the latest training statistics to be added to
                self.record.
            epoch: current epoch.
            total_epochs: total #epochs. Used for printing only.
            next_ckpt_epoch: Next epoch for evaluation and checkpointing. Used
                for printing only.
        """

        if self.log_file is not None:
            if not self._header_logged:
                self._header_logged = True
                self.log_file.write('epoch\t' + '\t'.join(new_record.keys()) + '\n')
            self.log_file.write(f'{epoch}\t' + '\t'.join(map(str, new_record.values())) + '\n')
        for key, val in new_record.items():
            print(f'{key}: {val:{self.fmt}}', end='\t')
            self.record[key].append(val)
        print(f'Epoch {int(epoch):5d}/{total_epochs:5d}\tNext ckpt: {next_ckpt_epoch:7d}', end='\r', flush=True)

    def log_and_clear_record(self) -> None:
        """Logs record to logger and reset self.record."""

        for key, val in self.record.items():
            _logger.info(f'{key:12s}: {np.mean(val):{self.fmt}}')
        self.record = defaultdict(list)

    def __del__(self) -> None:
        if self.log_file is not None:
            self.log_file.close()


def train_test_split(
    adata: anndata.AnnData,
    test_ratio: float = 0.1,
    seed: int = 1
) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """Splits the adata into a training set and a test set.

    Args:
        adata: the dataset to be splitted.
        test_ratio: ratio of the test data in adata.
        seed: random seed.

    Returns:
        the training set and the test set, both in AnnData format.
    """

    rng = np.random.default_rng(seed=seed)
    test_indices = rng.choice(adata.n_obs, size=int(test_ratio * adata.n_obs), replace=False)
    train_indices = list(set(range(adata.n_obs)).difference(test_indices))
    train_adata = adata[adata.obs_names[train_indices], :]
    test_adata = adata[adata.obs_names[test_indices], :]
    _logger.info(f'Keeping {test_adata.n_obs} cells ({test_ratio:g}) as test data.')
    return train_adata, test_adata


@log_arguments
def prepare_for_transfer(
    model: scETM,
    tgt_dataset: anndata.AnnData,
    aligned_src_genes: Sequence[str],
    keep_tgt_unique_genes: bool = False,
    fix_shared_genes: bool = False,
    batch_col: Union[str, None] = "batch_indices"
) -> Tuple[scETM, anndata.AnnData]:
    """Prepares the model (trained on the source dataset) and target dataset
    for transfer learning.

    The source and target datasets need to have shared genes for knowledge
    transfer to be possible.

    Args:
        model: an scETM model trained on the source dataset.
        tgt_dataset: the target dataset.
        aligned_src_genes: a list of source genes aligned to tgt_dataset. For
            example, if the source dataset is from mouse and the target from
            human, the caller should convert the mouse genes to the homologous
            human genes before passing the gene list here.
        keep_tgt_unique_genes: whether to keep target genes not found in the
            source dataset. If False, filter out all target-unique genes.
        fix_shared_genes: whether to fix the parameters of the input/output
            layer related to the shared genes.
        batch_col: a key in tgt_dataset.obs to the batch column.
    
    Returns:
        The transfered model and the prepared target dataset.
    """

    assert pd.Series(aligned_src_genes).is_unique, 'aligned_src_genes is not unique'
    assert tgt_dataset.var_names.is_unique, 'tgt_dataset.var_names is not unique'
    assert batch_col is None or batch_col in tgt_dataset.obs, f'{batch_col} not in tgt_dataset.obs'

    tgt_genes = tgt_dataset.var_names
    shared_genes = set(aligned_src_genes).intersection(tgt_genes)
    src_shared_indices = [i for i, gene in enumerate(aligned_src_genes) if gene in shared_genes]
    shared_genes = list(shared_genes)

    if not keep_tgt_unique_genes:
        tgt_dataset = tgt_dataset[:, shared_genes]
    else:
        tgt_indices = shared_genes + [gene for gene in tgt_genes if gene not in shared_genes]
        tgt_dataset = tgt_dataset[:, tgt_indices]
    if fix_shared_genes:
        n_fixed_genes = len(shared_genes)
    else:
        n_fixed_genes = 0
    tgt_model = copy.deepcopy(model)
    tgt_model.n_fixed_genes = n_fixed_genes
    tgt_model.n_trainable_genes = tgt_dataset.n_vars - n_fixed_genes
    tgt_model.n_batches = tgt_dataset.obs[batch_col].nunique() if batch_col is not None else 1

    # initialize rho_trainable_emb, batch and global bias
    tgt_model._init_encoder_first_layer()
    tgt_model._init_rho_trainable_emb()
    tgt_model._init_batch_and_global_biases()

    with torch.no_grad():
        rho_trainable_emb = model.rho_trainable_emb.get_param()
        if tgt_model.n_fixed_genes > 0:
            # initialize first layer
            tgt_model.q_delta[0].fixed.weight = nn.Parameter(model.q_delta[0].weight[:, src_shared_indices].detach())

            # model has trainable emb dim > 0 (shape of rho: [L_t, G])
            if tgt_model.trainable_gene_emb_dim > 0:
                # fix embeddings of shared genes: [L_t, G_s]
                tgt_model.rho_trainable_emb.fixed = rho_trainable_emb[:, src_shared_indices].detach().to(tgt_model.device)
        else:
            # initialize first layer
            tgt_model.q_delta[0].weight[:, :len(shared_genes)] = model.q_delta[0].weight[:, src_shared_indices].detach()

            # model has trainable emb dim > 0 (shape of rho: [L_t, G])
            if tgt_model.trainable_gene_emb_dim > 0:
                tgt_model.rho_trainable_emb.trainable[:, :len(shared_genes)] = rho_trainable_emb[:, src_shared_indices].detach()
            
        # model has fixed emb dim > 0 (shape of rho: [L_f, G])
        if model.rho_fixed_emb is not None:
            tgt_model.rho_fixed_emb = torch.zeros((model.rho_fixed_emb.size(0), tgt_dataset.n_vars), dtype=torch.float, device=tgt_model.device)
            tgt_model.rho_fixed_emb[:, :len(shared_genes)] = model.rho_fixed_emb[:, src_shared_indices].detach()  # [L_f, G_s]

    tgt_model = tgt_model.to(tgt_model.device)

    return tgt_model, tgt_dataset


def set_seed(seed: int) -> None:
    """Sets the random seed to seed.

    Args:
        seed: the random seed.
    """

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    _logger.info(f'Set seed to {seed}.')
