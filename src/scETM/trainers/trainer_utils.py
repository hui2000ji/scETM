import os
import copy
import pandas as pd
import random
from typing import DefaultDict, IO, List, Sequence, Union, Tuple
import logging
from collections import defaultdict

import numpy as np
import anndata
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from scETM.models import scETM
from scETM.logging_utils import log_arguments

_logger = logging.getLogger(__name__)


class _stats_recorder:
    """A utility class for recording training statistics.

    Attributes:
        record: the training statistics record.
        fmt: print format for the training statistics.
        log_file: the file stream to write logs to.
        writer: an initialized SummaryWriter for tensorboard logging.
    """

    def __init__(self,
        record_log_path: Union[str, None] = None,
        fmt: str = "10.4g",
        writer: Union[None, SummaryWriter] = None,
        metadata: Union[None, pd.DataFrame] = None
    ) -> None:
        """Initializes the statistics recorder.
        
        Args:
            record_log_path: the file path to write logs to.
            fmt: print format for the training statistics.
            tensorboard_dir: directory path to tensorboard logs. If None, do
                not log.
        """

        self.record: DefaultDict[List] = defaultdict(list)
        self.fmt: str = fmt
        self.log_file: Union[None, IO] = None
        self.writer: Union[None, SummaryWriter] = writer
        if writer is not None:
            metadata.to_csv(os.path.join(writer.get_logdir(), 'metadata.tsv'), sep='\t')
        if record_log_path is not None:
            self.log_file = open(record_log_path, 'w')
            self._header_logged: bool = False

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
            if self.writer is not None:
                self.writer.add_scalar(key, val, epoch)
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
