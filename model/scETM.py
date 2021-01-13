import anndata
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from scipy.sparse import csr_matrix
from .BaseCellModel import BaseCellModel

class scETM(BaseCellModel):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(adata, args)

        self.normalize_beta = args.model == 'ETM'
        self.n_topics = args.n_topics
        self.normed_loss = args.normed_loss
        self.norm_cells = args.norm_cells
        self.is_sparse = isinstance(adata.X, csr_matrix)
        if self.is_sparse:
            self.library_size = adata.X.sum(1)
        else:
            self.library_size = adata.X.sum(1, keepdims=True) 
 
        self.q_delta = self.get_fully_connected_layers(
            n_input=self.n_genes + ((self.n_batches - 1) if self.input_batch_id else 0),
            hidden_sizes=args.hidden_sizes,
            args=args
        )
        hidden_dim = args.hidden_sizes[-1]
        self.mu_q_delta = nn.Linear(hidden_dim, self.n_topics, bias=True)
        self.logsigma_q_delta = nn.Linear(hidden_dim, self.n_topics, bias=True)

        self.supervised = args.max_supervised_weight > 0
        if self.supervised:
            self.n_labels = adata.obs.cell_types.nunique()
            self.cell_type_clf = self.get_fully_connected_layers(self.n_topics, self.n_labels, args)

        self.rho_fixed, self.rho = None, None
        if 'gene_emb' in adata.varm:
            rho_fixed = adata.varm['gene_emb'].T  # L x G
            rho_fixed_std = rho_fixed.std(1, keepdims=True)
            rho_fixed_std[rho_fixed_std == 0.] = 1
            rho_fixed = (rho_fixed - rho_fixed.mean(1, keepdims=True)) / rho_fixed_std
            self.rho_fixed = torch.FloatTensor(rho_fixed).to(device=device)
            if self.trainable_gene_emb_dim:
                self.rho = nn.Parameter(torch.randn(self.trainable_gene_emb_dim, self.n_genes))
        else:
            self.rho = nn.Parameter(torch.randn(self.trainable_gene_emb_dim, self.n_genes))

        self.alpha = nn.Parameter(torch.randn(self.n_topics, self.trainable_gene_emb_dim + (adata.varm['gene_emb'].shape[1] if self.rho_fixed is not None else 0)))
        if self.batch_scaling:
            self.gene_bias = nn.Parameter(torch.randn(self.n_batches, self.n_genes))
        
        self.global_bias = nn.Parameter(torch.randn(1, self.n_genes)) if args.global_bias else None

        ## cap log variance within [-10, 10]
        self.max_logsigma = 10
        self.min_logsigma = -10

        # self.init_emb()

    @staticmethod
    def _init_emb(emb):
        if isinstance(emb, nn.Linear):
            nn.init.xavier_uniform_(emb.weight.data)
            if emb.bias is not None:
                emb.bias.data.fill_(0.0)
        elif isinstance(emb, nn.Sequential):
            for child in emb:
                scETM._init_emb(child)

    def init_emb(self):
        for m in self.modules():
            self._init_emb(m)

    def _get_batch_indices_oh(self, data_dict):
        if 'batch_indices_oh' in data_dict:
            w_batch_id = data_dict['batch_indices_oh']
        else:
            batch_indices = data_dict['batch_indices'].unsqueeze(1)
            w_batch_id = torch.zeros((batch_indices.shape[0], self.n_batches), dtype=torch.float32, device=self.device)
            w_batch_id.scatter_(1, batch_indices, 1.)
            w_batch_id = w_batch_id[:, :self.n_batches - 1]
            data_dict['batch_indices_oh'] = w_batch_id
        return w_batch_id

    def get_cell_emb_weights(self):
        return super().get_cell_emb_weights(['theta', 'delta'])

    def forward(self, data_dict, hyper_param_dict=dict(val=True)):
        cells, library_size = data_dict['cells'], data_dict['library_size']
        normed_cells = cells / library_size if self.norm_cells else cells

        if self.input_batch_id:
            normed_cells = torch.cat((normed_cells, self._get_batch_indices_oh(data_dict)), dim=1)
        
        q_delta = self.q_delta(normed_cells)
        mu_q_delta = self.mu_q_delta(q_delta)

        if 'val' in hyper_param_dict:
            theta = F.softmax(mu_q_delta, dim=-1)
            if self.supervised:
                cell_type_logit = self.cell_type_clf(mu_q_delta)
                return dict(theta=theta, delta=mu_q_delta, cell_type_logit=cell_type_logit)
            else:
                return dict(theta=theta, delta=mu_q_delta)

        logsigma_q_delta = self.logsigma_q_delta(q_delta).clamp(self.min_logsigma, self.max_logsigma)
        q_delta = Independent(Normal(
            loc=mu_q_delta,
            scale=logsigma_q_delta.exp()
        ), 1)

        delta = q_delta.rsample()
        theta = F.softmax(delta, dim=-1)  # [batch_size, n_topics]

        rhos = [param for param in (self.rho_fixed, self.rho) if param is not None]
        rho = torch.cat(rhos, dim=0) if len(rhos) > 1 else rhos[0]
        beta = self.alpha @ rho

        if self.normalize_beta:
            recon = torch.mm(theta, F.softmax(beta, dim=-1)) + 1e-30
            nll = (-recon.log() * self.mask_gene_expression(normed_cells if self.normed_loss else cells)).sum(-1).mean()
        else:
            recon_logit = torch.mm(theta, beta)  # [batch_size, n_genes]
            if self.global_bias is not None:
                recon_logit += self.global_bias
            if self.batch_scaling:
                recon_logit += self.gene_bias[data_dict['batch_indices']]
            nll = (-F.log_softmax(recon_logit, dim=-1) * self.mask_gene_expression(normed_cells if self.normed_loss else cells)).sum(-1).mean()

        kl_delta = self.get_kl(mu_q_delta, logsigma_q_delta).mean()
        loss = nll + hyper_param_dict['beta'] * kl_delta
        tracked_items = dict(loss=loss, nll=nll, kl_delta=kl_delta)
        if self.supervised:
            cell_type_logit = self.cell_type_clf(delta)
            cross_ent = F.cross_entropy(cell_type_logit, data_dict['cell_type_indices']).mean()
            loss += hyper_param_dict['supervised_weight'] * cross_ent
            tracked_items['cross_ent'] = cross_ent

        tracked_items = {k: v.detach().item() for k, v in tracked_items.items()}

        fwd_dict = dict(
            theta=theta,
            delta=delta
        )
        
        return loss, fwd_dict, tracked_items
