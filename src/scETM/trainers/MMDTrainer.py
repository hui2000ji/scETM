from typing import Mapping, Union
import numpy as np
from torch import optim
import torch
import logging
import anndata
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.autograd as autograd
import torch.nn.functional as F

from scETM.models import BaseCellModel, BatchClassifier
from scETM.logging_utils import log_arguments
from .UnsupervisedTrainer import UnsupervisedTrainer

_logger = logging.getLogger(__name__)


class MMDTrainer(UnsupervisedTrainer):
    """Docstring here (TODO)
    """

    @log_arguments
    def __init__(self,
        model: BaseCellModel,
        adata: anndata.AnnData,
        ckpt_dir: Union[str, None] = None,
        test_ratio: float = 0.,
        data_split_seed: int = 1,
        init_lr: float = 5e-3,
        lr_decay: float = 5e-5,
        batch_size: int = 2000,
        train_instance_name: str = "scETMbatch",
        restore_epoch: int = 0,
        seed: int = -1,
    ) -> None:
        """Docstring here (TODO)
        """

        super().__init__(model,
            adata,
            ckpt_dir=ckpt_dir,
            test_ratio=test_ratio,
            data_split_seed=data_split_seed,
            init_lr=init_lr,
            lr_decay=lr_decay,
            batch_size=batch_size,
            train_instance_name=train_instance_name,
            restore_epoch=restore_epoch,
            seed=seed
        )

        if restore_epoch > 0 and type(self) == MMDTrainer:
            self.ckpt_dir = ckpt_dir
            self.load_ckpt(restore_epoch, self.ckpt_dir)

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
        mmd_warmup_ratio: float = 1/6,
        min_mmd_weight: float = 0.,
        max_mmd_weight: float = 1,
        **train_kwargs
    ) -> None:
        train_kwargs.update(dict(
            mmd_warmup_ratio=mmd_warmup_ratio,
            min_mmd_weight=min_mmd_weight,
            max_mmd_weight=max_mmd_weight,
        ))
        return super().train(n_epochs=n_epochs,
            eval_every=eval_every,
            n_samplers=n_samplers,
            kl_warmup_ratio=kl_warmup_ratio,
            min_kl_weight=min_kl_weight,
            max_kl_weight=max_kl_weight,
            eval=eval,
            batch_col=batch_col,
            save_model_ckpt=save_model_ckpt,
            record_log_path=record_log_path,
            writer=writer,
            eval_result_log_path=eval_result_log_path,
            eval_kwargs=eval_kwargs,
            **train_kwargs
        )
    
    def do_train_step(self, dataloader, **kwargs) -> Mapping[str, torch.Tensor]:
        """Docstring (TODO)
        """

        # construct hyper_param_dict
        hyper_param_dict = {
            'kl_weight': self._calc_weight(self.epoch, kwargs['n_epochs'], 0, kwargs['kl_warmup_ratio'], kwargs['min_kl_weight'], kwargs['max_kl_weight']),
            'mmd_weight': self._calc_weight(self.epoch, kwargs['n_epochs'], 0, kwargs['mmd_warmup_ratio'], kwargs['min_mmd_weight'], kwargs['max_mmd_weight'])
        }

        # construct data_dict
        data_dict = {k: v.to(self.device) for k, v in next(dataloader).items()}

        # train for one step, record tracked items (e.g. loss)
        self.model.train()
        if hyper_param_dict['mmd_weight'] > 0.:
            def loss_update_callback(loss, fwd_dict, new_record):
                mmd_loss = mmd(fwd_dict[self.model.clustering_input], data_dict['batch_indices'], self.model.n_batches)
                loss += mmd_loss * hyper_param_dict['mmd_weight']
                new_record['mmd_loss'] = mmd_loss.detach().item()
                return loss, new_record
        else:
            loss_update_callback = None
        new_record = self.model.train_step(self.optimizer, data_dict, hyper_param_dict, loss_update_callback)

        return new_record, hyper_param_dict


def partition(data, partitions, num_partitions):
    res = []
    partitions = partitions.flatten()
    for i in range(num_partitions):
        indices = torch.nonzero((partitions == i), as_tuple=False).squeeze(1)
        res += [data[indices]]
    return res


def pairwise_squared_euclidean(x: torch.Tensor, y: torch.Tensor):
    return ((x.unsqueeze(-1) - y.T) ** 2).sum(1)


def gaussian_kernel_matrix(x, y, alphas):
    """Computes multiscale-RBF kernel between x and y.

        Parameters
        ----------
        x: torch.Tensor
            Tensor with shape [batch_size, z_dim].
        y: torch.Tensor
            Tensor with shape [batch_size, z_dim].
        alphas: Tensor

        Returns
        -------
        Returns the computed multiscale-RBF kernel between x and y.
    """

    dist = pairwise_squared_euclidean(x, y).contiguous()
    beta = 1. / (2. * alphas.view(-1, 1))

    s = beta @ dist.view(1, -1)
    return torch.exp(-s).sum(0).view_as(dist)


def mmd_loss_calc(source_features, target_features):
    """Initializes Maximum Mean Discrepancy(MMD) between source_features and target_features.

        - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.

        Parameters
        ----------
        source_features: torch.Tensor
            Tensor with shape [batch_size, z_dim]
        target_features: torch.Tensor
            Tensor with shape [batch_size, z_dim]

        Returns
        -------
        Returns the computed MMD between x and y.
    """
    alphas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    alphas = torch.tensor(alphas, device=source_features.device)

    cost = gaussian_kernel_matrix(source_features, source_features, alphas).mean()
    cost += gaussian_kernel_matrix(target_features, target_features, alphas).mean()
    cost -= 2 * gaussian_kernel_matrix(source_features, target_features, alphas).mean()

    return cost


def mmd(
    embedding: torch.Tensor,
    batch_indices: torch.Tensor,
    n_batches: int,
    n_old_batches: Union[int, None] = None
) -> torch.Tensor:
    """Initializes Maximum Mean Discrepancy(MMD) between every different condition.

        Parameters
        ----------
        n_batches: integer
            Number of batches the data contain.
        n_old_batches: integer
            If not 'None', mmd loss is only calculated on #new batches.
        embedding: torch.Tensor
            Torch Tensor of computed latent data.
        c: torch.Tensor
            Torch Tensor of condition labels.

        Returns
        -------
        Returns MMD loss.
    """

    # partition separates y into num_cls subsets w.r.t. their labels c
    emb_partitions = partition(embedding, batch_indices, n_batches)
    loss = 0.
    if n_old_batches is not None:
        for i in range(n_old_batches):
            for j in range(n_old_batches, n_batches):
                if emb_partitions[i].size(0) < 2 or emb_partitions[j].size(0) < 2:
                    continue
                loss += mmd_loss_calc(emb_partitions[i], emb_partitions[j])
    else:
        for i in range(len(emb_partitions)):
            if emb_partitions[i].size(0) < 1:
                continue
            for j in range(i):
                if emb_partitions[j].size(0) < 1:
                    continue
                loss += mmd_loss_calc(emb_partitions[i], emb_partitions[j])

    return loss
