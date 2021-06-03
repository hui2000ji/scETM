from typing import Mapping, Union
import numpy as np
from torch import optim
import torch
import logging
import anndata
from torch.utils.tensorboard import SummaryWriter

from scETM.models import BaseCellModel, BatchClassifier
from scETM.logging_utils import log_arguments
from .UnsupervisedTrainer import UnsupervisedTrainer

_logger = logging.getLogger(__name__)


class BatchAdversarialTrainer(UnsupervisedTrainer):
    """Docstring here (TODO)
    """

    attr_fname: Mapping[str, str] = dict(
        model = 'model',
        optimizer = 'opt',
        batch_clf = 'bmodel',
        batch_clf_optimizer = 'bopt'
    )

    @log_arguments
    def __init__(self,
        model: BaseCellModel,
        batch_clf: BatchClassifier,
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
        batch_clf_init_lr: float = 1e-3,
        batch_clf_lr_decay: float = 5e-5,
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

        # instantiate the batch classifier
        self.batch_clf: BatchClassifier = batch_clf

        # instantiate the batch classifier optimizer
        self.batch_clf_init_lr = batch_clf_init_lr
        self.batch_clf_lr = batch_clf_init_lr
        self.batch_clf_lr_decay = batch_clf_lr_decay
        self.batch_clf_optimizer: optim.Optimizer = optim.Adam(self.batch_clf.parameters(), lr=self.batch_clf_init_lr)

        if restore_epoch > 0 and type(self) == BatchAdversarialTrainer:
            self.ckpt_dir = ckpt_dir
            self.load_ckpt(restore_epoch, self.ckpt_dir)
    
    def update_step(self, jump_to_step: Union[None, int] = None) -> None:
        """Docstring here (TODO)
        """

        super().update_step(jump_to_step=jump_to_step)
        if self.batch_clf_lr_decay:
            if jump_to_step is None:
                self.batch_clf_lr *= np.exp(-self.batch_clf_lr_decay)
            else:
                self.batch_clf_lr = self.batch_clf_init_lr * np.exp(-jump_to_step * self.batch_clf_lr_decay)
            for param_group in self.batch_clf_optimizer.param_groups:
                param_group['lr'] = self.batch_clf_lr

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
        clf_cutoff_ratio: float = 1/6,
        clf_warmup_ratio: float = 1/3,
        min_clf_weight: float = 0.,
        max_clf_weight: float = 0.005,
        g_steps: int = 1,
        d_steps: int = 8,
        **train_kwargs
    ) -> None:
        """Trains the model, optionally evaluates performance and logs results.

        See `UnsupervisedTrainer.train` for common argument docstrings.

        Args:
            clf_cutoff_ratio: ratio of n_epochs where the classifier weight is
                zero.
            clf_warmup_ratio: ratio of classifier term warmup epochs and
                n_epochs.
            min_clf_weight: minimum weight of the classifier term.
            max_clf_weight: maximum weight of the classifier term.
            g_steps: times to update scETM in a training step.
            d_steps: times to update the batch classifier in a training step.
            train_kwargs: other kwargs to pass to self.do_train_step().
        """

        train_kwargs.update(dict(
            clf_cutoff_ratio=clf_cutoff_ratio,
            clf_warmup_ratio=clf_warmup_ratio,
            min_clf_weight=min_clf_weight,
            max_clf_weight=max_clf_weight,
            g_steps=g_steps,
            d_steps=d_steps
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
            'clf_weight': self._calc_weight(self.epoch, kwargs['n_epochs'], kwargs['clf_cutoff_ratio'], kwargs['clf_warmup_ratio'], kwargs['min_clf_weight'], kwargs['max_clf_weight'])
        }

        # construct data_dict
        data_dict = {k: v.to(self.device) for k, v in next(dataloader).items()}

        # train for one step, record tracked items (e.g. loss)
        self.model.train()
        self.batch_clf.train()
        for _ in range(kwargs['g_steps']):
            if hyper_param_dict['clf_weight'] > 0.:
                def loss_update_callback(loss, fwd_dict, new_record):
                    _, fwd_dict, _ = self.batch_clf(fwd_dict[self.model.clustering_input], data_dict['batch_indices'])
                    model_loss = fwd_dict['model_loss']
                    loss += model_loss * hyper_param_dict['clf_weight']
                    new_record['model_loss'] = model_loss.detach().item()
                    return loss, new_record
            else:
                loss_update_callback = None
            new_record = self.model.train_step(self.optimizer, data_dict, hyper_param_dict, loss_update_callback)

        _, fwd_dict, _ = self.model(data_dict, hyper_param_dict)
        emb = fwd_dict[self.model.clustering_input].detach()
        for _ in range(kwargs['d_steps']):
            clf_record = self.batch_clf.train_step(self.batch_clf_optimizer, emb, data_dict['batch_indices'])
        new_record.update(clf_record)

        return new_record, hyper_param_dict

    def before_eval(self,
        batch_col: str,
        **kwargs
    ) -> None:
        """Docstring (TODO)
        """

        adata = self.adata
        assert adata.n_vars == self.model.n_fixed_genes + self.model.n_trainable_genes
        logits = []
        embs = []

        self.batch_clf.eval()
        def store_emb_and_nll(data_dict, fwd_dict):
            emb = fwd_dict[self.model.clustering_input]
            logits.append(self.batch_clf(emb, data_dict['batch_indices'])['logit'].detach().cpu())
            embs.append(emb.detach().cpu())

        self.model._apply_to(adata, batch_col, self.batch_size, dict(decode=False), callback=store_emb_and_nll)

        embs = torch.cat(embs, dim=0)
        logits = torch.cat(logits, dim=0)
        pred = logits.argmax(-1)
        correct = pred.numpy() == adata.obs[batch_col].astype('category').cat.codes

        adata.obsm[self.model.clustering_input] = embs.numpy()
        adata.obs['clf_pred'] = adata.obs[batch_col].astype('category').cat.categories[np.array(pred)]
        adata.obs['clf_pred'] = adata.obs['clf_pred'].astype('category')
        adata.obs['clf_correct'] = np.array(correct, dtype='str')
        adata.obs['clf_correct'] = adata.obs['clf_correct'].astype('category')
        if 'color_by' not in adata.uns:
            adata.uns['color_by'] = {'clf_pred', 'clf_correct'}
        else:
            adata.uns['color_by'] |= {'clf_pred', 'clf_correct'}
