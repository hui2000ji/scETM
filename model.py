import scanpy as sc
import anndata
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.autograd import Function
from scvi.models.distributions import ZeroInflatedNegativeBinomial
from scvi.models.modules import DecoderSCVI
from torch.distributions import Normal, Independent
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, \
    normalized_mutual_info_score


class BaseCellModel(nn.Module):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.gene_emb_dim = args.gene_emb_dim
        if not args.no_eval:
            self.n_labels = adata.obs.cell_types.nunique()
        self.device = device
        self.n_cells = adata.n_obs
        self.n_genes = adata.n_vars
        self.n_batches = adata.obs.batch_indices.nunique()
        self.batch_removal = args.input_batch_id
        self.batch_scaling = args.batch_scaling
        self.batch_size = args.batch_size
        self.mask_ratio = args.mask_ratio
        if self.mask_ratio < 0 or self.mask_ratio > 0.5:
            raise ValueError("Mask ratio should be between 0 and 0.5.")
        if 'condition' in adata.obs and 'condition' not in args.always_draw:
            args.always_draw.append('condition')

        self.is_sparse = isinstance(adata.X, csr_matrix)
        if self.is_sparse:
            self.library_size = adata.X.sum(1)
        else:
            self.library_size = adata.X.sum(1, keepdims=True)
        self.X = adata.X
        self.batch_indices = adata.obs.batch_indices.astype(int)

    def mask_gene_expression(self, cells):
        if self.mask_ratio > 0:
            return cells * (torch.rand_like(cells, device=self.device, dtype=torch.float32) * (1 - 2 * self.mask_ratio))
        else:
            return cells

    def get_cell_emb_weights(self, weight_names):
        self.eval()
        if isinstance(weight_names, str):
            weight_names = [weight_names]

        if self.n_cells > self.batch_size:
            weights = {name: [] for name in weight_names}
            for start in range(0, self.n_cells, self.batch_size):
                X = self.X[start: start + self.batch_size, :]
                if self.is_sparse:
                    X = X.todense()
                cells = torch.FloatTensor(X).to(self.device)
                library_size = torch.FloatTensor(self.library_size[start: start + self.batch_size]).to(self.device)
                data_dict = dict(cells=cells, library_size=library_size,
                    cell_indices=torch.arange(start, min(start + self.batch_size, self.n_cells), device=self.device))
                if self.batch_removal or self.batch_scaling:
                    batch_indices = torch.LongTensor(self.batch_indices[start: start + self.batch_size]).to(self.device)
                    data_dict['batch_indices'] = batch_indices
                fwd_dict = self(data_dict, dict(val=True))
                for name in weight_names:
                    weights[name].append(fwd_dict[name].detach().cpu())
            weights = {name: torch.cat(weights[name], dim=0).numpy() for name in weight_names}
        else:
            X = self.X.todense() if self.is_sparse else self.X
            cells = torch.FloatTensor(X).to(self.device)
            library_size = torch.FloatTensor(self.library_size).to(self.device)
            data_dict = dict(cells=cells, library_size=library_size, cell_indices=torch.arange(self.n_cells, device=self.device))
            if self.batch_removal or self.batch_scaling:
                batch_indices = torch.LongTensor(self.batch_indices).to(self.device)
                data_dict['batch_indices'] = batch_indices
            fwd_dict = self(data_dict, dict(val=True))
            weights = {name: fwd_dict[name].detach().cpu().numpy() for name in weight_names}
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


class MixtureOfMultinomial(BaseCellModel):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(adata, args)

        args.tracked_metric = "q_nmi"

        self.norm_cells = args.norm_cells
        self.normed_loss = args.normed_loss
        self.batch_scaling = args.batch_scaling
        self.is_sparse = isinstance(adata.X, csr_matrix)
        if self.is_sparse:
            self.library_size = adata.X.sum(1)
        else:
            self.library_size = adata.X.sum(1, keepdims=True)
        self.gene_decoder = nn.Parameter(torch.randn(self.n_labels, self.n_genes, device=device))

        self.E_result = None

    def init_emb(self):
        for m in self.modules():
            self._init_emb(m)

    def get_cell_emb_weights(self):
        return {}

    def _get_batch_indices_oh(self, data_dict):
        if 'batch_indices_oh' in data_dict:
            w_batch_id = data_dict['batch_indices_oh']
        else:
            batch_indices = data_dict['batch_indices'].unsqueeze(1)
            w_batch_id = torch.zeros((batch_indices.shape[0], self.n_batches), dtype=torch.float32, device=self.device)
            w_batch_id.scatter_(1, batch_indices, 1.)
            data_dict['batch_indices_oh'] = w_batch_id
        return w_batch_id

    def forward(self, data_dict, hyper_param_dict=dict(E=True)):
        cells, library_size = data_dict['cells'], data_dict['library_size']
        norm_cells = cells / library_size if self.norm_cells else cells

        x = F.log_softmax(self.gene_decoder, dim=-1).unsqueeze(0)
        log_pxz = (x * (norm_cells if self.normed_loss else cells).unsqueeze(1)).sum(-1)

        # [batch_size, n_labels]
        if hyper_param_dict['E']:
            qz = F.softmax(log_pxz, dim=-1).detach()
            self.E_result = qz
        else:
            qz = self.E_result
        fwd_dict = dict(log_pxz=log_pxz, qz=qz)
        return fwd_dict

    def get_cell_type(self, cell_gene_sampler, adata=None, args=None, hard_accum=False):
        self.eval()
        result = dict(
            q_cell_type=self.get_cell_type('qz').argmax(-1),
        )
        return result

    def get_loss(self, fwd_dict, data_dict, hyper_param_dict):
        log_pxz, qz = fwd_dict['log_pxz'], fwd_dict['qz']

        loss = -(qz * log_pxz).sum(1).mean()
        new_items = {}
        return loss, new_items


class MixtureOfZINB(BaseCellModel):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(adata, args)

        args.tracked_metric = "q_nmi"

        self.norm_cells = args.norm_cells
        self.normed_loss = args.normed_loss
        self.batch_scaling = args.batch_scaling
        self.is_sparse = isinstance(adata.X, csr_matrix)
        if self.is_sparse:
            self.library_size = adata.X.sum(1)
        else:
            self.library_size = adata.X.sum(1, keepdims=True)
        self.batch_indices = adata.obs.batch_indices.astype(int)

        self.mu_decoder = nn.Parameter(torch.randn(self.n_labels, self.n_genes, device=device))
        self.logdisp_decoder = nn.Parameter(torch.randn(self.n_labels, self.n_genes, device=device))
        self.zi_decoder = nn.Parameter(torch.randn(self.n_labels, self.n_genes, device=device))

        self.E_result = None

    def init_emb(self):
        for m in self.modules():
            self._init_emb(m)

    def get_cell_emb_weights(self):
        return {}

    def _get_batch_indices_oh(self, data_dict):
        if 'batch_indices_oh' in data_dict:
            w_batch_id = data_dict['batch_indices_oh']
        else:
            batch_indices = data_dict['batch_indices'].unsqueeze(1)
            w_batch_id = torch.zeros((batch_indices.shape[0], self.n_batches), dtype=torch.float32, device=self.device)
            w_batch_id.scatter_(1, batch_indices, 1.)
            data_dict['batch_indices_oh'] = w_batch_id
        return w_batch_id

    def forward(self, data_dict, hyper_param_dict=dict(E=True)):
        cells, library_size = data_dict['cells'], data_dict['library_size']
        norm_cells = cells / library_size if self.norm_cells else cells

        mu_x = F.softmax(self.mu_decoder, dim=-1).unsqueeze(0) * library_size.reshape(library_size.shape[0], 1, 1)
        disp_x = self.logdisp_decoder.exp().unsqueeze(0)
        zi_logits = self.zi_decoder.unsqueeze(0)

        # [batch_size, n_labels]
        log_pxz = ZeroInflatedNegativeBinomial(mu=mu_x, theta=disp_x, zi_logits=zi_logits).log_prob((norm_cells if self.normed_loss else cells).unsqueeze(1)).sum(-1)
        if hyper_param_dict['E']:
            qz = F.softmax(log_pxz, dim=-1).detach()
            self.E_result = qz
        else:
            qz = self.E_result
        fwd_dict = dict(log_pxz=log_pxz, qz=qz)
        return fwd_dict

    def get_cell_type(self, cell_gene_sampler, adata=None, args=None, hard_accum=False):
        self.eval()
        result = dict(
            q_cell_type=self.get_cell_type('qz').argmax(-1),
        )
        return result

    def get_loss(self, fwd_dict, data_dict, hyper_param_dict):
        log_pxz, qz = fwd_dict['log_pxz'], fwd_dict['qz']

        loss = -(qz * log_pxz).sum(1).mean()
        new_items = {}
        return loss, new_items


class scETM(BaseCellModel):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(adata, args)

        self.n_topics = args.n_topics
        self.normed_loss = args.normed_loss
        self.norm_cells = args.norm_cells
        self.is_sparse = isinstance(adata.X, csr_matrix)
        if self.is_sparse:
            self.library_size = adata.X.sum(1)
        else:
            self.library_size = adata.X.sum(1, keepdims=True) 
 
        self.q_delta = self.get_fully_connected_layers(
            n_input=self.n_genes + ((self.n_batches - 1) if self.batch_removal else 0),
            hidden_sizes=args.hidden_sizes,
            args=args
        )
        hidden_dim = args.hidden_sizes[-1]
        self.mu_q_delta = nn.Linear(hidden_dim, self.n_topics, bias=True)
        self.logsigma_q_delta = nn.Linear(hidden_dim, self.n_topics, bias=True)

        self.supervised = args.max_supervised_weight > 0
        if self.supervised:
            self.cell_type_clf = self.get_fully_connected_layers(self.n_topics, self.n_labels, args)

        # self.rho = nn.Linear(self.gene_emb_dim, self.n_genes, bias=False)
        self.rho_fixed, self.rho = None, None
        if 'gene_emb' in adata.varm:
            self.rho_fixed = torch.FloatTensor(adata.varm['gene_emb'].T).to(device=device)
            if self.gene_emb_dim:
                self.rho = nn.Parameter(torch.randn(self.gene_emb_dim, self.n_genes))
        else:
            self.rho = nn.Parameter(torch.randn(self.gene_emb_dim, self.n_genes))

        self.alpha = nn.Parameter(torch.randn(self.n_topics, self.gene_emb_dim + (adata.varm['gene_emb'].shape[1] if self.rho_fixed is not None else 0)))
        if self.batch_scaling:
            self.gene_bias = nn.Parameter(torch.randn(self.n_batches, self.n_genes))

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
        batch_size = cells.shape[0]

        if self.batch_removal:
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

        if self.supervised:
            cell_type_logit = self.cell_type_clf(delta)

        # beta = self.rho(self.alpha)  # [n_topics, n_genes]
        rhos = [param for param in (self.rho_fixed, self.rho) if param is not None]
        rho = torch.cat(rhos, dim=0) if len(rhos) > 1 else rhos[0]
        beta = self.alpha @ rho

        # recon_logit = self.beta(theta)  # [batch_size, n_genes]
        recon_logit = torch.mm(theta, beta)  # [batch_size, n_genes]

        if self.batch_scaling:
            # recon_logit += self.cell_bias[data_dict['cell_indices']] * self.gene_bias[data_dict['batch_indices']]
            recon_logit += self.gene_bias[data_dict['batch_indices']]
        # recon_logit = torch.mm(theta, F.softmax(self.beta, dim=-1))
        fwd_dict = dict(
            nll=(-F.log_softmax(recon_logit, dim=-1) * self.mask_gene_expression(normed_cells if self.normed_loss else cells)).sum(-1).mean(),
            # nll=((F.softmax(recon_logit, dim=-1) - ((cells / library_size) if not self.norm_cells else cells)) ** 2).sum(-1).mean(),
            # nll=(-recon_logit.log() * cells).sum(-1).mean(),
            kl_delta=get_kl(mu_q_delta, logsigma_q_delta).mean(),
            theta=theta,
            delta=delta
        )
        if self.supervised:
            fwd_dict['cross_ent'] = F.cross_entropy(cell_type_logit, data_dict['cell_type_indices']).mean()
        
        return fwd_dict

    def get_cell_type(self, cell_gene_sampler, adata=None, args=None, hard_accum=False):
        self.eval()
        weights = self.get_cell_emb_weights()
        p_cell_type = weights['delta'].argmax(-1)

        adata.obsm['delta'] = weights['delta']
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep='delta')

        louvain_cell_type, louvain_metadata = get_louvain_type(self, adata, args, use_rep='delta')
        leiden_cell_type, leiden_metadata = get_leiden_type(self, adata, args, use_rep='delta')
        result = dict(
            p_cell_type=p_cell_type,
            louvain_cell_type=louvain_cell_type,
            leiden_cell_type=leiden_cell_type
        )
        return result, dict(
            louvain=louvain_metadata,
            leiden=leiden_metadata
        )

    def get_loss(self, fwd_dict, data_dict, hyper_param_dict):
        nll, kl_delta = fwd_dict['nll'], fwd_dict['kl_delta']
        loss = nll + hyper_param_dict['beta'] * kl_delta
        tracked_items = dict(loss=loss, nll=nll, kl_delta=kl_delta)
        if self.supervised:
            cross_ent = fwd_dict['cross_ent']
            loss += hyper_param_dict['supervised_weight'] * cross_ent
            tracked_items['cross_ent'] = cross_ent
        tracked_items = {k: v.detach().item() for k, v in tracked_items.items()}
        return loss, tracked_items


class scETMMultiDecoder(BaseCellModel):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(adata, args)

        args.neg_samples = 0
        args.cell_sampling = True
        args.tracked_metric = "l_nmi"
        # args.always_draw = ('cell_types', 'leiden', 'p', 'batch_indices')

        self.n_topics = args.n_topics 
        self.normed_loss = args.normed_loss
        self.norm_cells = args.norm_cells
        self.group_by = args.group_by
        self.is_sparse = isinstance(adata.X, csr_matrix)
        if self.is_sparse:
            self.library_size = adata.X.sum(1)
        else:
            self.library_size = adata.X.sum(1, keepdims=True)
 
        self.q_delta = self.get_fully_connected_layers(
            n_input=self.n_genes + ((self.n_batches - 1) if self.batch_removal else 0),
            hidden_sizes=args.hidden_sizes,
            args=args
        )
        hidden_dim = args.hidden_sizes[-1]
        self.mu_q_delta = nn.Linear(hidden_dim, self.n_topics, bias=True)
        self.logsigma_q_delta = nn.Linear(hidden_dim, self.n_topics, bias=True)

        self.n_groups = adata.obs[self.group_by].nunique()
        self.group_col = torch.LongTensor(adata.obs[self.group_by].values.codes.copy()).to(device)
        self.beta = [nn.Linear(self.n_topics + ((self.n_batches - 1) if self.batch_removal else 0), self.n_genes, bias=False) for _ in range(self.n_groups)]
        for i, module in enumerate(self.beta):
            self.add_module(f'beta_{i}', module)

        if self.batch_scaling:
            # self.cell_bias = nn.Parameter(torch.ones(self.n_cells, 1))
            self.gene_bias = [nn.Parameter(torch.randn(self.n_batches, self.n_genes)) for _ in range(self.n_groups)]
            for i, module in enumerate(self.gene_bias):
                self.register_parameter(f'gene_bias_{i}', module)

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
                scETMMultiDecoder._init_emb(child)

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

    def forward(self, data_dict, hyper_param_dict=dict(E=True)):
        cells, library_size = data_dict['cells'], data_dict['library_size']
        normed_cells = cells / library_size if self.norm_cells else cells
        batch_size = cells.shape[0]
        cell_indices = data_dict['cell_indices']
        
        q_delta = self.q_delta(normed_cells)
        mu_q_delta = self.mu_q_delta(q_delta)

        if 'val' in hyper_param_dict:
            return dict(theta=F.softmax(mu_q_delta, dim=-1), delta=mu_q_delta)

        logsigma_q_delta = self.logsigma_q_delta(q_delta).clamp(self.min_logsigma, self.max_logsigma)
        q_delta = Independent(Normal(
            loc=mu_q_delta,
            scale=logsigma_q_delta.exp()
        ), 1)

        delta = q_delta.rsample()
        theta = F.softmax(delta, dim=-1)  # [batch_size, n_topics]

        # beta = self.rho(self.alpha)  # [n_topics, n_genes]
        if self.batch_removal:
            new_theta = torch.cat((theta, self._get_batch_indices_oh(data_dict)), dim=1)
        else:
            new_theta = theta

        recon_logit = torch.zeros(self.batch_size, self.n_genes, device=self.device, dtype=torch.float32)
        for i in range(self.n_groups):
            indices = self.group_col[cell_indices] == i
            recon_logit[indices] = self.beta[i](new_theta[indices])
            if self.batch_scaling:
                recon_logit[indices] += self.gene_bias[i][data_dict['batch_indices'][indices]]
        
        fwd_dict = dict(
            nll=(-F.log_softmax(recon_logit, dim=-1) * (norm_cells if self.normed_loss else cells)).sum(-1).mean(),
            # nll=((F.softmax(recon_logit, dim=-1) - ((cells / library_size) if not self.norm_cells else cells)) ** 2).sum(-1).mean(),
            # nll=(-recon_logit.log() * cells).sum(-1).mean(),
            kl_delta=get_kl(mu_q_delta, logsigma_q_delta).mean(),
            theta=theta,
            delta=delta
        )
        return fwd_dict

    def get_cell_type(self, cell_gene_sampler, adata=None, args=None, hard_accum=False):
        self.eval()
        weights = self.get_cell_emb_weights()
        p_cell_type = weights['delta'].argmax(-1)

        adata.obsm['delta'] = weights['delta']
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep='delta')

        louvain_cell_type, louvain_metadata = get_louvain_type(self, adata, args, use_rep='delta')
        leiden_cell_type, leiden_metadata = get_leiden_type(self, adata, args, use_rep='delta')
        result = dict(
            p_cell_type=p_cell_type,
            louvain_cell_type=louvain_cell_type,
            leiden_cell_type=leiden_cell_type
        )
        return result, dict(
            louvain=louvain_metadata,
            leiden=leiden_metadata
        )

    def get_loss(self, fwd_dict, data_dict, hyper_param_dict):
        nll, kl_delta = fwd_dict['nll'], fwd_dict['kl_delta']
        loss = nll + hyper_param_dict['beta'] * kl_delta
        tracked_items = dict(loss=loss, nll=nll, kl_delta=kl_delta)
        tracked_items = {k: v.detach().item() for k, v in tracked_items.items()}
        return loss, tracked_items


class SupervisedClassifier(BaseCellModel):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(adata, args)
        self.norm_cells = args.norm_cells
        self.is_sparse = isinstance(adata.X, csr_matrix)
        if self.is_sparse:
            self.library_size = adata.X.sum(1)
        else:
            self.library_size = adata.X.sum(1, keepdims=True)
        self.batch_indices = adata.obs.batch_indices.astype(int)
        self.encoder = self.get_fully_connected_layers(
            n_input=self.n_genes + ((self.n_batches - 1) if self.batch_removal else 0),
            hidden_sizes=args.hidden_sizes,
            args=args
        )
        hidden_dim = args.hidden_sizes[-1]
        self.clf = nn.Linear(hidden_dim, self.n_labels, bias=True)
    
    def forward(self, data_dict, hyper_param_dict):
        cells, library_size = data_dict['cells'], data_dict['library_size']
        normed_cells = cells / library_size if self.norm_cells else cells
        batch_size = cells.shape[0]
        
        embeddings = self.encoder(normed_cells)
        logit = self.clf(embeddings)
        return dict(logit=logit)
    
    def get_loss(self, fwd_dict, data_dict, hyper_param_dict):
        logit = fwd_dict['logit']
        cross_entropy = F.cross_entropy(logit, data_dict['cell_type_indices']).mean()
        tracked_items = {'cross_ent': cross_entropy}
        tracked_items = {k: v.detach().item() for k, v in tracked_items.items()}
        return cross_entropy, tracked_items

    def get_cell_emb_weights(self):
        return super().get_cell_emb_weights(['logit'])

    def get_cell_type(self, cell_gene_sampler, adata=None, args=None, hard_accum=False):
        self.eval()
        weights = self.get_cell_emb_weights()
        p_cell_type = weights['logit'].argmax(-1)

        adata.obsm['logit'] = weights['logit']
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep='logit')

        louvain_cell_type, louvain_metadata = get_louvain_type(self, adata, args, use_rep='logit')
        leiden_cell_type, leiden_metadata = get_leiden_type(self, adata, args, use_rep='logit')
        result = dict(
            p_cell_type=p_cell_type,
            louvain_cell_type=louvain_cell_type,
            leiden_cell_type=leiden_cell_type
        )
        return result, dict(
            louvain=louvain_metadata,
            leiden=leiden_metadata
        )


def get_louvain_type(model, adata, args, use_rep='w_cell_emb'):
    print('Performing louvain clustering...', flush=True, end='\r')
    aris = []  # List of (resolution, ARI score)
    print('=' * 10 + ' louvain ' + '=' * 10)
    for res in args.louvain_resolutions:
        col = f'louvain_{res}'
        sc.tl.louvain(adata, resolution=res, key_added=col)
        ari = adjusted_rand_score(adata.obs.cell_types, adata.obs[col])
        n_unique = adata.obs[col].nunique()
        aris.append((res, ari, n_unique))
        print(f'Resolution: {res:5.3g}\tARI: {ari:7.4f}\t# labels: {n_unique}')

    aris.sort(key=lambda x: x[1], reverse=True)
    best_res = aris[0][0]
    best_ari = aris[0][1]

    aris.sort(key=lambda x: x[2])
    n_labels = adata.obs.cell_types.nunique()
    if len(aris) > 2 and not args.fix_resolutions:
        if aris[1][2] < n_labels / 2 and aris[-1][2] <= n_labels:
            args.louvain_resolutions = [res + 0.05 for res in args.louvain_resolutions]
        elif aris[-1][2] > n_labels and args.louvain_resolutions[0] > 0.01:
            args.louvain_resolutions = [res - min(0.1, min(args.louvain_resolutions) / 2) for res in args.louvain_resolutions]
 
    return adata.obs[f'louvain_{best_res}'].astype(int), dict(ari=best_ari, res=best_res)


def get_leiden_type(model, adata, args, use_rep='w_cell_emb'):
    print('Performing leiden clustering...', flush=True, end='\r')
    aris = []  # List of (resolution, ARI score)

    print('=' * 10 + ' leiden ' + '=' * 10)
    for res in args.leiden_resolutions:
        col = f'leiden_{res}'
        sc.tl.leiden(adata, resolution=res, key_added=col)
        ari = adjusted_rand_score(adata.obs.cell_types, adata.obs[col])
        n_unique = adata.obs[col].nunique()
        aris.append((res, ari, n_unique))
        print(f'Resolution: {res:5.3g}\tARI: {ari:7.4f}\t# labels: {n_unique}')

    aris.sort(key=lambda x: x[1], reverse=True)
    best_res = aris[0][0]
    best_ari = aris[0][1]

    aris.sort(key=lambda x: x[2])
    n_labels = adata.obs.cell_types.nunique()
    if len(aris) > 2 and not args.fix_resolutions:
        if aris[1][2] < n_labels / 2 and aris[-1][2] <= n_labels:
            args.leiden_resolutions = [res + 0.05 for res in args.leiden_resolutions]
        elif aris[-1][2] > n_labels and args.leiden_resolutions[0] > 0.01:
            args.leiden_resolutions = [res - min(0.1, min(args.leiden_resolutions) / 2) for res in args.leiden_resolutions]

    return adata.obs[f'leiden_{best_res}'].astype(int), dict(ari=best_ari, res=best_res)


def get_k_type(model):
    print('Performing k-means clustering...', flush=True, end='\r')
    kmeans = KMeans(n_clusters=model.n_labels, n_init=20)
    arr = model.get_cell_emb_weights()['w_cell_emb']
    kmeans.fit_transform(arr)
    kmeans_cell_type = kmeans.labels_
    return kmeans_cell_type


def get_kl(mu, logsigma):
    return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)


class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass
        return method