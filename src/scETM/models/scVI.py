from typing import Sequence, Union
import anndata
import numpy as np
from scipy.sparse.base import spmatrix
import torch
from torch import autograd
import logging
from torch.distributions import kl_divergence
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, NegativeBinomial

from scETM.logging_utils import log_arguments
from .BaseCellModel import BaseCellModel
from .model_utils import (
    get_fully_connected_layers,
    get_kl
)

_logger = logging.getLogger(__name__)


class scVI(BaseCellModel):

    emb_names = ['z', 's']
    clustering_input = 'z'
    max_logsigma = 10
    min_logsigma = -10
    
    @log_arguments
    def __init__(self,
        n_trainable_genes: int,
        n_batches: int,
        n_fixed_genes: int = 0,
        n_topics: int = 50,
        hidden_sizes: Sequence[int] = (128,),
        bn: bool = True,
        dropout_prob: float = 0.1,
        norm_cells: bool = True,
        normed_loss: bool = False,
        reconstruction_loss: str = "nb",
        input_batch_id: bool = False,
        enable_batch_specific_dispersion: bool = True,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__(n_trainable_genes, n_batches, n_fixed_genes, need_batch=n_batches > 1 and (input_batch_id or enable_batch_specific_dispersion), device=device)

        self.n_topics: int = n_topics
        self.hidden_sizes: Sequence[int] = hidden_sizes
        self.bn: bool = bn
        self.dropout_prob: float = dropout_prob
        self.norm_cells: bool = norm_cells
        self.normed_loss: bool = normed_loss
        self.input_batch_id: bool = input_batch_id
        self.reconstruction_loss: str = reconstruction_loss
        self.enable_batch_specific_dispersion = enable_batch_specific_dispersion
        if self.n_batches <= 1:
            _logger.warning(f'n_batches == {self.n_batches}, disabling batch bias')
            self.input_batch_id = False

        self.encoder = get_fully_connected_layers(
            n_trainable_input=self.n_genes + ((self.n_batches - 1) if self.input_batch_id else 0),
            hidden_sizes=hidden_sizes,
            n_trainable_output=self.n_topics * 4,
            bn=bn,
            dropout_prob=dropout_prob
        )
        
        hidden_sizes = list(hidden_sizes).copy()
        hidden_sizes.reverse()
        
        self.decoder = get_fully_connected_layers(
            n_trainable_input=self.n_topics * 2 + ((self.n_batches - 1) if self.input_batch_id else 0),
            hidden_sizes=hidden_sizes,
            n_trainable_output=self.n_genes,
            bn=bn,
            dropout_prob=dropout_prob
        )

        if self.reconstruction_loss == "mse":
            self.recon_batch_clf = get_fully_connected_layers(
                n_trainable_input=self.n_genes,
                hidden_sizes=self.hidden_sizes,
                n_trainable_output=n_batches,
                bn=bn,
                dropout_prob=dropout_prob
            )
        else:
            if enable_batch_specific_dispersion:
                self.px_total_count = nn.Parameter(torch.randn(self.n_batches, self.n_genes))
            else:
                self.px_total_count = nn.Parameter(torch.randn(1, self.n_genes))
            
        self.to(device)

    def decode(self, z, s, data_dict):
        decoder_inputs = [z, s]
        if self.input_batch_id:
            decoder_inputs.append(self._get_batch_indices_oh(data_dict))
        decoder_input = torch.cat(decoder_inputs, dim=-1)
        px_logits = self.decoder(decoder_input)

        if self.reconstruction_loss != "mse":
            if self.enable_batch_specific_dispersion:
                px_total_count = self.px_total_count[data_dict['batch_indices']].clamp(self.min_logsigma, self.max_logsigma).exp()
            else:
                px_total_count = self.px_total_count[torch.zeros(size=(z.size(0),), dtype=torch.long, device=z.device)].clamp(self.min_logsigma, self.max_logsigma).exp()
        else:
            px_total_count = None
        return px_total_count, px_logits

    def forward(self, data_dict, hyper_param_dict=dict(val=True)):
        cells, library_size = data_dict['cells'], data_dict['library_size']
        normed_cells = cells / library_size
        cells_for_loss = normed_cells if self.normed_loss else cells
        input_cells = normed_cells if self.norm_cells else cells
        if self.input_batch_id:
            input_cells = torch.cat((input_cells, self._get_batch_indices_oh(data_dict)), dim=1)

        mu_qz, logsigma_qz, mu_qs, logsigma_qs = self.encoder(input_cells).chunk(4, dim=-1)
        qz = Independent(Normal(
            loc=mu_qz,
            scale=logsigma_qz.clamp(self.min_logsigma, self.max_logsigma).exp()
        ), 1)
        z = qz.rsample()
        qs = Independent(Normal(
            loc=mu_qs,
            scale=logsigma_qs.clamp(self.min_logsigma, self.max_logsigma).exp()
        ), 1)
        s = qs.rsample()

        if not self.training:
            total_count, logits = self.decode(mu_qz, mu_qs, data_dict)
            fwd_dict = dict(
                z=mu_qz,
                s=mu_qs,
                total_count=total_count,
                logits=logits,
                nll = self.get_reconstruction_loss(cells_for_loss, total_count, logits).sum()
            )
            return fwd_dict

        if self.reconstruction_loss == 'mse':
            perm_mask = np.arange(cells.size(0), dtype=np.int64)
            np.random.shuffle(perm_mask)
            perm_mask = torch.LongTensor(perm_mask)
            _, perm_logits = self.decode(z, s[perm_mask, :], data_dict)
            perm_pred_logit = self.recon_batch_clf(perm_logits.softmax(dim=-1))
            perm_ce = F.cross_entropy(perm_pred_logit, data_dict['batch_indices'][perm_mask])


        total_count, logits = self.decode(z, s, data_dict)
        nll = self.get_reconstruction_loss(cells_for_loss, total_count, logits).mean()
        kl_z = get_kl(mu_qz, logsigma_qz).mean()
        kl_s = get_kl(mu_qs, logsigma_qs).mean()
        loss = nll + hyper_param_dict['kl_weight'] * kl_z + hyper_param_dict['kl_weight'] * kl_s
        if self.reconstruction_loss == 'mse':
            loss += 0.2 * perm_ce
            record = dict(loss=loss, nll=nll, kl_z=kl_z, kl_s=kl_s, perm_ce=perm_ce)
        else:
            record = dict(loss=loss, nll=nll, kl_z=kl_z, kl_s=kl_s, perm_ce=perm_ce)

        record = {k: v.detach().item() for k, v in record.items()}

        fwd_dict = dict(
            z=z,
            s=s,
            total_count=total_count,
            logits=logits
        )
        
        return loss, fwd_dict, record

    def sample_x(self, total_count, logits) -> torch.Tensor:
        # Reconstruction Loss
        if self.reconstruction_loss == "nb":
            x = Independent(NegativeBinomial(total_count=total_count, logits=logits), 1).sample()
        else:
            raise NotImplementedError
        return x

    def get_reconstruction_loss(self, x, total_count, logits) -> torch.Tensor:
        """Return the reconstruction loss (for a minibatch)
        """
        # Reconstruction Loss
        if self.reconstruction_loss == "nb":
            reconst_loss = -Independent(NegativeBinomial(total_count=total_count, logits=logits, validate_args=False), 1).log_prob(x).mean()
        elif self.reconstruction_loss == "mse":
            reconst_loss = F.mse_loss(logits.softmax(dim=-1), x, reduction='none').sum(-1).mean()
        else:
            raise NotImplementedError
        return reconst_loss
