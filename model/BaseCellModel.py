import anndata
import torch
import torch.nn as nn
from batch_sampler import CellSampler


class BaseCellModel(nn.Module):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.trainable_gene_emb_dim = args.trainable_gene_emb_dim
        self.device = device
        self.n_cells = adata.n_obs
        self.n_genes = adata.n_vars
        self.n_batches = adata.obs.batch_indices.nunique()
        self.input_batch_id = args.input_batch_id
        self.batch_scaling = args.batch_scaling
        self.batch_size = args.batch_size
        self.mask_ratio = args.mask_ratio
        if self.mask_ratio < 0 or self.mask_ratio > 0.5:
            raise ValueError("Mask ratio should be between 0 and 0.5.")

        self.adata, self.args = adata, args

    def mask_gene_expression(self, cells):
        if self.mask_ratio > 0:
            return cells * (torch.rand_like(cells, device=self.device, dtype=torch.float32) * (1 - 2 * self.mask_ratio))
        else:
            return cells

    def get_cell_emb_weights(self, weight_names):
        self.eval()
        if isinstance(weight_names, str):
            weight_names = [weight_names]

        sampler = CellSampler(self.adata, self.args, n_epochs=1, shuffle=False)
        weights = {name: [] for name in weight_names}
        for data_dict in sampler:
            data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
            fwd_dict = self(data_dict, dict(val=True))
            for name in weight_names:
                weights[name].append(fwd_dict[name].detach().cpu())
        weights = {name: torch.cat(weights[name], dim=0).numpy() for name in weight_names}
        return weights

    @staticmethod
    def get_fully_connected_layers(n_input, hidden_sizes, args):
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        layers = []
        for size in hidden_sizes:
            layers.append(nn.Linear(n_input, size))
            layers.append(nn.ReLU())
            if not args.no_bn:
                layers.append(nn.BatchNorm1d(size))
            if args.dropout_prob:
                layers.append(nn.Dropout(args.dropout_prob))
            n_input = size
        return nn.Sequential(*layers)
    
    @staticmethod
    def get_kl(mu, logsigma):
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)
