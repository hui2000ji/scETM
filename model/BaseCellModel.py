import anndata
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCellModel(nn.Module):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.trainable_gene_emb_dim = args.trainable_gene_emb_dim
        self.device = device
        self.n_genes = adata.n_vars
        self.n_batches = adata.obs.batch_indices.nunique()
        self.input_batch_id = args.input_batch_id
        self.batch_scaling = args.batch_scaling
        self.batch_size = args.batch_size
        self.mask_ratio = args.mask_ratio
        if self.mask_ratio < 0 or self.mask_ratio > 0.5:
            raise ValueError("Mask ratio should be between 0 and 0.5.")

        self.args = args

    def mask_gene_expression(self, cells):
        if self.mask_ratio > 0:
            return F.dropout(cells, p=self.mask_ratio, training=self.training)
        else:
            return cells

    @staticmethod
    def get_fully_connected_layers(n_input, hidden_sizes, n_output=None, bn=True, drop_prob=0., bn_track_running_stats=True):
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        layers = []
        for size in hidden_sizes:
            layers.append(nn.Linear(n_input, size))
            layers.append(nn.ReLU())
            if bn:
                layers.append(nn.BatchNorm1d(size, track_running_stats=bn_track_running_stats))
            if drop_prob:
                layers.append(nn.Dropout(drop_prob))
            n_input = size
        if n_output is not None:
            layers.append(nn.Linear(n_input, n_output))
        return nn.Sequential(*layers)
    
    @staticmethod
    def get_kl(mu, logsigma):
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)
