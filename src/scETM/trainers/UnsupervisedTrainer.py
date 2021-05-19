import os
from pathlib import Path
from matplotlib.figure import Figure
import time
from typing import Mapping, Union
import psutil
import logging

import numpy as np
import anndata
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from scETM.batch_sampler import CellSampler, MultithreadedCellSampler
from scETM.eval_utils import evaluate
from scETM.models import BaseCellModel, scETM
from scETM.logging_utils import initialize_logger, log_arguments
from .trainer_utils import train_test_split, set_seed, _stats_recorder

_logger = logging.getLogger(__name__)


class UnsupervisedTrainer:
    """Unsupervised trainer for single-cell modeling.

    Sets up the random seed, dataset split, optimizer and logger, and executes
    training and evaluation loop.

    Attributes:
        attr_fname: a dict mapping attributes of the trainer (a model or an
            optimizer) to file name prefixes of checkpoints.
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

    attr_fname: Mapping[str, str] = dict(
        model = 'model',
        optimizer = 'opt'
    )

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
        for attr, fname in self.attr_fname.items():
            fpath = os.path.join(ckpt_dir, f'{fname}-{restore_epoch}')
            getattr(self, attr).load_state_dict(torch.load(fpath))
        _logger.info(f'Parameters and optimizers restored from {ckpt_dir}.')
        initialize_logger(self.ckpt_dir)
        _logger.info(f'ckpt_dir: {self.ckpt_dir}')
        self.update_step(restore_epoch * self.steps_per_epoch)

    @staticmethod
    def _calc_weight(
        epoch: int,
        n_epochs: int,
        cutoff_ratio: float = 0.,
        warmup_ratio: float = 1/3,
        min_weight: float = 0.,
        max_weight: float = 1e-7
    ) -> float:
        """Calculates weights.

        Args:
            epoch: current epoch.
            n_epochs: the total number of epochs to train the model.
            cutoff_ratio: ratio of cutoff epochs (set weight to zero) and
                n_epochs.
            warmup_ratio: ratio of warmup epochs and n_epochs.
            min_weight: minimum weight.
            max_weight: maximum weight.

        Returns:
            The current weight of the KL term.
        """

        fully_warmup_epoch = n_epochs * warmup_ratio
        if cutoff_ratio > warmup_ratio:
            _logger.warning(f'Cutoff_ratio {cutoff_ratio} is bigger than warmup_ratio {warmup_ratio}. This may not be an expected behavior.')
        if epoch < n_epochs * cutoff_ratio:
            return 0.
        if warmup_ratio:
            return max(min(1., epoch / fully_warmup_epoch) * max_weight, min_weight)
        else:
            return max_weight

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
        writer: Union[None, SummaryWriter] = None,
        eval_result_log_path: Union[str, None] = None,
        eval_kwargs: Union[None, dict] = None,
        **train_kwargs
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
            record_log_path: file path to log the training records. If None, do
                not log.
            writer: an initialized SummaryWriter for tensorboard logging.
            eval_result_log_path: file path to log the evaluation results. If
                None, do not log.
            eval_kwargs: kwargs to pass to the evaluate function.
            train_kwargs: kwargs to pass to self.do_train_step().
        """

        default_eval_kwargs = dict(
            batch_col = batch_col,
            plot_fname = f'{self.train_instance_name}_{self.model.clustering_input}',
            plot_dir = self.ckpt_dir,
            writer = writer
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
        recorder = _stats_recorder(record_log_path=record_log_path, writer=writer, metadata=self.adata.obs)
        next_ckpt_epoch = min(int(np.ceil(self.epoch / eval_every) * eval_every), n_epochs)

        while self.epoch < n_epochs:
            new_record, hyper_param_dict = self.do_train_step(dataloader,
                n_epochs = n_epochs,
                kl_warmup_ratio = kl_warmup_ratio,
                min_kl_weight = min_kl_weight,
                max_kl_weight = max_kl_weight,
                **train_kwargs
            )
            recorder.update(new_record, self.epoch, n_epochs, next_ckpt_epoch)
            self.update_step()

            # log and evaluate
            if self.epoch >= next_ckpt_epoch or self.epoch >= n_epochs:
                _logger.info('=' * 10 + f'Epoch {next_ckpt_epoch:.0f}' + '=' * 10)

                # log memory cost
                _logger.info(repr(psutil.Process().memory_info()))
                # log current lr and kl_weight
                if self.lr_decay:
                    _logger.info(f'{"lr":12s}: {self.lr:12.4g}')
                for k, v in hyper_param_dict.items():
                    _logger.info(f'{k:12s}: {v:12.4g}')

                # log statistics of tracked items
                recorder.log_and_clear_record()
                if self.test_adata is not self.adata:
                    test_nll = self.model.get_cell_embeddings_and_nll(self.test_adata, self.batch_size, batch_col=batch_col, emb_names=[])
                    if test_nll is not None:
                        _logger.info(f'test nll: {test_nll:7.4f}')
                else:
                    test_nll = None
                
                if eval:
                    # get embeddings, evaluate and log results
                    current_eval_kwargs = eval_kwargs.copy()
                    current_eval_kwargs['plot_fname'] = current_eval_kwargs['plot_fname'] + f'_epoch{int(next_ckpt_epoch)}'
                    self.before_eval(batch_col=batch_col)
                    if isinstance(self.model, scETM):
                        self.model.write_topic_gene_embeddings_to_tensorboard(writer, self.adata.var_names, f'gene_topic_emb_epoch{int(next_ckpt_epoch)}')
                    result = evaluate(adata = self.adata, embedding_key = self.model.clustering_input, **current_eval_kwargs)
                    result['test_nll'] = test_nll
                    self._log_eval_result(result, next_ckpt_epoch, writer, eval_result_log_path)

                if next_ckpt_epoch and save_model_ckpt and self.ckpt_dir is not None:
                    # checkpointing
                    self.save_model_and_optimizer(next_ckpt_epoch)

                _logger.info('=' * 10 + f'End of evaluation' + '=' * 10)
                next_ckpt_epoch = min(eval_every + next_ckpt_epoch, n_epochs)

        del recorder
        _logger.info("Optimization Finished: %s" % self.ckpt_dir)
        if isinstance(sampler, MultithreadedCellSampler):
            sampler.join(0.1)

    def save_model_and_optimizer(self, next_ckpt_epoch: int) -> None:
        """Docstring (TODO)
        """

        for attr, fname in self.attr_fname.items():
            torch.save(getattr(self, attr).state_dict(), os.path.join(self.ckpt_dir, f'{fname}-{next_ckpt_epoch}'))

    def _log_eval_result(self,
        result: Mapping[str, Union[float, None, Figure]],
        next_ckpt_epoch: int,
        writer: Union[None, SummaryWriter],
        eval_result_log_path: Union[str, None] = None
    ) -> None:
        """Docstring (TODO)
        """
        if writer is not None:
            for k, v in result.items():
                if isinstance(v, float):
                    writer.add_scalar(k, v, next_ckpt_epoch)
        if eval_result_log_path is not None:
            with open(eval_result_log_path, 'a+') as f:
                # ckpt_dir, epoch, test_nll, ari, nmi, k_bet, ebm, time, seed
                f.write(f'{Path(self.ckpt_dir).name}\t'
                        f'{next_ckpt_epoch}\t'
                        f'{result["test_nll"]}\t'
                        f'{result["ari"]}\t'
                        f'{result["nmi"]}\t'
                        f'{result["k_bet"]}\t'
                        f'{result["ebm"]}\t'
                        f'{time.strftime("%m_%d-%H_%M_%S")}\t'
                        f'{self.seed}\n')

    def do_train_step(self, dataloader, **kwargs) -> Mapping[str, torch.Tensor]:
        """Docstring (TODO)
        """

        # construct hyper_param_dict
        hyper_param_dict = {
            'kl_weight': self._calc_weight(self.epoch, kwargs['n_epochs'], 0, kwargs['kl_warmup_ratio'], kwargs['min_kl_weight'], kwargs['max_kl_weight'])
        }

        # construct data_dict
        data_dict = {k: v.to(self.device) for k, v in next(dataloader).items()}

        # train for one step, record tracked items (e.g. loss)
        new_record = self.model.train_step(self.optimizer, data_dict, hyper_param_dict)

        return new_record, hyper_param_dict

    def before_eval(self, batch_col: str, **kwargs) -> None:
        """Docstring (TODO)
        """

        self.model.get_cell_embeddings_and_nll(self.adata, self.batch_size, batch_col=batch_col, emb_names=[self.model.clustering_input])

