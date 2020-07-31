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


def get_emb(num_emb, emb_dim, n_enc_layers):
    if n_enc_layers == 1:
        return nn.Embedding(num_emb, emb_dim)
    enc_layers = [nn.Embedding(num_emb, emb_dim)]
    for _ in range(n_enc_layers - 1):
        if _ > 0:
            enc_layers.append(nn.ReLU())
        enc_layers.append(nn.Linear(emb_dim, emb_dim))
    return nn.Sequential(*enc_layers)


def get_dec(num_emb, emb_dim, n_dec_layers):
    if n_dec_layers == 1:
        return nn.Linear(emb_dim, num_emb)
    dec_layers = []
    for _ in range(n_dec_layers - 1):
        dec_layers.append(nn.Linear(emb_dim, emb_dim))
        dec_layers.append(nn.ReLU())
    dec_layers.append(nn.Linear(emb_dim, num_emb))
    return nn.Sequential(*dec_layers)


class CellGeneModel(nn.Module):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.emb_dim = args.emb_dim
        self.n_labels = args.n_labels
        self.device = device
        self.n_cells = adata.n_obs
        self.n_genes = adata.n_vars
        self.n_batches = adata.obs.batch_indices.nunique()
        self.cell_type_dec = args.cell_type_dec
        self.neg_samples = args.neg_samples
        self.batch_size = args.batch_size
        self.eval_batches = args.eval_batches
        self.encoder_depth = args.encoder_depth
        self.decoder_depth = args.decoder_depth
        self.cell_batch_scaling = args.cell_batch_scaling

        if self.cell_batch_scaling:
            self.batch_emb = get_emb(self.n_batches, self.emb_dim, self.encoder_depth).to(device)

        self.cell_type_emb = nn.Linear(
            self.emb_dim, self.n_labels, bias=False).to(device)
        if self.cell_type_dec:
            self.cell_type_dec = nn.Linear(
                self.n_labels, self.emb_dim).to(device)
        self.w_cell_emb = get_emb(
            self.n_cells, self.emb_dim, self.encoder_depth).to(device)
        self.w_gene_emb = get_emb(
            self.n_genes, self.emb_dim, self.encoder_depth).to(device)

        if self.neg_samples:
            self.c_gene_emb = nn.Embedding(
                self.n_genes, self.emb_dim).to(device)
            if self.decoder_depth > 1:
                self.gene_decoder = get_dec(
                    self.emb_dim, self.emb_dim,
                    self.decoder_depth - 1).to(device)
        else:
            self.gene_decoder = get_dec(
                self.n_genes, self.emb_dim, self.decoder_depth).to(device)
        self.init_emb()

    @staticmethod
    def _init_emb(emb):
        if isinstance(emb, nn.Linear):
            nn.init.xavier_uniform_(emb.weight.data)
            if emb.bias is not None:
                emb.bias.data.fill_(0.0)
        elif isinstance(emb, nn.Sequential):
            for child in emb:
                CellGeneModel._init_emb(child)

    def init_emb(self):
        for m in self.modules():
            self._init_emb(m)

    def get_cell_emb_weights(self):
        if self.encoder_depth == 1:
            return {'w_cell_emb': self.w_cell_emb.weight.detach().cpu().numpy()}
        elif self.encoder_depth > 1:
            all_cells = torch.arange(
                0, self.n_cells, 1, dtype=torch.long, device=self.device)
            w_emb = self.w_cell_emb(all_cells).detach().cpu().numpy()
            return {'w_cell_emb': w_emb}

    def _get_batch_indices_oh(self, data_dict):
        if 'batch_indices_oh' in data_dict:
            w_batch_id = data_dict['batch_indices_oh']
        else:
            batch_indices = data_dict['batch_indices'].unsqueeze(1)
            w_batch_id = torch.zeros((batch_indices.shape[0], self.n_batches), dtype=torch.float32, device=self.device)
            w_batch_id.scatter_(1, batch_indices, 1.)
            data_dict['batch_indices_oh'] = w_batch_id
        return w_batch_id

    def forward(self, data_dict, hyper_param_dict=dict(tau=1.)):
        cells, genes = data_dict['cells'], data_dict['genes']
        tau = hyper_param_dict['tau']

        w_cell_emb = self.w_cell_emb(cells)
        w_gene_emb = self.w_gene_emb(genes)

        pz_logit = self.cell_type_emb(w_cell_emb)
        qz_logit = self.cell_type_emb(w_cell_emb * w_gene_emb)

        if self.training:
            qz = F.gumbel_softmax(logits=qz_logit, tau=tau, hard=True)
        else:
            tmp = qz_logit.argmax(dim=-1, keepdims=True)
            qz = torch.zeros(qz_logit.shape).to(
                self.device).scatter_(1, tmp, 1.)

        recon_c_emb = torch.mm(qz, self.cell_type_emb.weight)
        if self.cell_batch_scaling:
            batch_indices = data_dict['batch_indices']
            recon_c_emb = recon_c_emb + self.batch_emb(batch_indices)
        if self.neg_samples and self.decoder_depth > 1:
            recon_c_emb = self.gene_decoder(recon_c_emb)
        fwd_dict = {
            "qz": F.softmax(qz_logit, dim=-1),
            "pz": F.softmax(pz_logit, dim=-1),
            "w_emb": w_cell_emb
        }
        if self.neg_samples:
            fwd_dict["recon_c_emb"] = recon_c_emb
        else:
            recon_c_logit = self.gene_decoder(recon_c_emb)
            fwd_dict["recon_c_gene"] = recon_c_logit
        return fwd_dict

    def get_cell_type(self, cell_gene_sampler, adata=None, args=None, hard_accum=False):
        self.eval()
        return dict(
            p_cell_type=get_p_type(self, adata),
            q_cell_type=get_q_type(
                self, cell_gene_sampler, hard_accum),
            ouvain_cell_type=get_louvain_type(self, adata, args),
            leiden_cell_type=get_leiden_type(self, adata, args),
        )

    def _get_c2g_loss(self, fwd_dict, data_dict, hyper_param_dict):
        genes = data_dict['genes']
        return F.cross_entropy(
            fwd_dict['recon_c_gene'],
            genes,
            reduction='sum'
        ) / genes.shape[0]

    def _get_c2g_loss_neg(self, fwd_dict, data_dict, hyper_param_dict,
                          emb_name):
        # self.cell_scaler[cells] * self.gene_scaler[batch_indices]
        cells, genes, neg_genes = data_dict['cells'], data_dict['genes'], data_dict['neg_genes']

        c_gene_emb = self.c_gene_emb(genes)
        c_gene_emb_neg = self.c_gene_emb(neg_genes) * (-1)
        recon_c_gene_emb = fwd_dict[emb_name]

        pos = F.logsigmoid((c_gene_emb * recon_c_gene_emb).sum(dim=-1))
        neg = F.logsigmoid(
            (c_gene_emb_neg * recon_c_gene_emb.unsqueeze(1)).sum(dim=-1)
        ).sum(dim=-1)

        return (pos + hyper_param_dict['neg_weight'] * neg).mean() * (-1.)

    def get_c2g_loss(self, fwd_dict, data_dict, hyper_param_dict):
        if self.neg_samples:
            return self._get_c2g_loss_neg(
                fwd_dict, data_dict, hyper_param_dict, 'recon_c_emb')
        else:
            return self._get_c2g_loss(fwd_dict, data_dict, hyper_param_dict)

    def get_LINE_loss(self, fwd_dict, data_dict, hyper_param_dict):
        if self.neg_samples:
            return self._get_c2g_loss_neg(
                fwd_dict, data_dict, hyper_param_dict, 'w_emb')
        else:
            genes = data_dict['genes']
            recon_c_gene_LINE = self.gene_decoder(fwd_dict['w_emb'])
            BCE_gene_LINE = F.cross_entropy(
                recon_c_gene_LINE, genes, reduction='sum') / genes.shape[0]
            return BCE_gene_LINE

    def get_loss(self, fwd_dict, data_dict, hyper_param_dict):
        qz, pz = fwd_dict['qz'], fwd_dict['pz']
        cells = data_dict['cells']

        qz_plus = qz + 1e-30
        log_qz = torch.log(qz_plus)
        # KLD = torch.sum(qz * (log_qz - torch.log(pz + 1e-20)), dim=-1).mean()
        KLD = (log_qz - torch.log(pz + 1e-30)).sum(-1).mean()

        BCE_gene = self.get_c2g_loss(fwd_dict, data_dict, hyper_param_dict)
        loss = BCE_gene + hyper_param_dict['beta'] * KLD
        new_items = {'KL': KLD, 'c2g': BCE_gene}

        if hyper_param_dict['epsilon']:
            BCE_gene_LINE = self.get_LINE_loss(
                fwd_dict, data_dict, hyper_param_dict)
            loss = loss + hyper_param_dict['epsilon'] * BCE_gene_LINE
            new_items['Llg'] = BCE_gene_LINE

        # word reconstruction loss from qz
        if hyper_param_dict['zeta']:
            recon_w_cell = torch.mm(qz, self.cell_type_emb.weight)
            BCEw_cell = F.cross_entropy(
                recon_w_cell, cells, reduction='sum') / cells.shape[0]
            BCEw = self.g2c_factor * BCEw_cell

            loss = loss + hyper_param_dict['zeta'] * BCEw
            new_items['Lwc'] = BCEw_cell

        loss, new_items = get_gg_cc_loss(
            self, fwd_dict, data_dict, hyper_param_dict, loss, new_items)
        return loss, new_items


class vGraphEM(nn.Module):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        if sum([elem in args.always_draw for elem in ['cell_types', 'q', 'batch_indices']]) == 3 and len(args.always_draw) == 3:
            # args.always_draw = ['p', 'ouvain', 'leiden', 'cell_types']
            args.always_draw = ['p', 'cell_types']
        args.neg_samples = 0

        if args.m_step <= 0:
            raise ValueError('m-step should > 0 in EM models')

        self.emb_dim = args.emb_dim
        self.n_labels = args.n_labels
        self.device = device
        self.n_cells = adata.n_obs
        self.n_genes = adata.n_vars
        self.n_batches = adata.obs.batch_indices.nunique()
        self.neg_samples = args.neg_samples
        self.batch_size = args.batch_size
        self.eval_batches = args.eval_batches
        self.encoder_depth = args.encoder_depth
        self.decoder_depth = args.decoder_depth
        self.batch_removal = args.max_lambda > 0
        self.cell_batch_scaling = args.cell_batch_scaling

        self.cell_type_emb = nn.Linear(
            self.emb_dim + (self.n_batches if self.batch_removal else 0), self.n_labels, bias=False).to(device)
        self.w_cell_emb = get_emb(
            self.n_cells, self.emb_dim, self.encoder_depth).to(device)
        if self.cell_batch_scaling:
            # self.cell_scaler = nn.Parameter(torch.ones((self.n_cells, 1), dtype=torch.float32)).to(device)
            self.gene_scaler = nn.Parameter(torch.rand((self.n_batches, self.n_genes), dtype=torch.float32)).to(device)

        self.gene_decoder = get_dec(
            self.n_genes, self.emb_dim + (self.n_batches if self.batch_removal else 0), self.decoder_depth).to(device)
        self.init_emb()

        self.E_result = None

    @staticmethod
    def _init_emb(emb):
        if isinstance(emb, nn.Linear):
            nn.init.xavier_uniform_(emb.weight.data)
            if emb.bias is not None:
                emb.bias.data.fill_(0.0)
        elif isinstance(emb, nn.Sequential):
            for child in emb:
                vGraphEM._init_emb(child)

    def init_emb(self):
        for m in self.modules():
            self._init_emb(m)

    def get_cell_emb_weights(self):
        if self.encoder_depth == 1 and isinstance(self.w_cell_emb, nn.Linear):
            return {'w_cell_emb': self.w_cell_emb.weight.detach().cpu().numpy()}
        else:
            all_cells = torch.arange(
                0, self.n_cells, 1, dtype=torch.long, device=self.device)
            w_emb = self.w_cell_emb(all_cells).detach().cpu().numpy()
            return {'w_cell_emb': w_emb}

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
        cells, genes = data_dict['cells'], data_dict['genes']
        cell_type_emb_weight = self.cell_type_emb.weight[:, :self.emb_dim] \
            if self.batch_removal else self.cell_type_emb.weight

        all_cells = torch.arange(
            0, self.n_cells, 1, dtype=torch.long, device=self.device)
        w_cell_emb = self.w_cell_emb(all_cells)[cells]
        if self.batch_removal:
            w_cell_emb = torch.cat((w_cell_emb, self._get_batch_indices_oh(data_dict)), dim=1)
        pz_w = F.softmax(self.cell_type_emb(w_cell_emb), dim=-1)

        if self.cell_batch_scaling:
            batch_indices = data_dict['batch_indices']
            c_logit = self.gene_decoder(cell_type_emb_weight) + self.gene_scaler[batch_indices] 
        else:
            c_logit = self.gene_decoder(cell_type_emb_weight)
        pc_z = F.softmax(c_logit, dim=-1).T

        pcz_w = pz_w * pc_z[genes]
        if hyper_param_dict['E']:
            qz_wc = (pcz_w / pcz_w.sum(1, keepdims=True)).detach()
            self.E_result = qz_wc
        else:
            qz_wc = self.E_result
        fwd_dict = dict(pcz_w=pcz_w, pz=pz_w, qz_wc=qz_wc)
        return fwd_dict

    def get_cell_type(self, cell_gene_sampler, adata=None, args=None, hard_accum=False):
        self.eval()
        result = dict(
            p_cell_type=get_p_type(self, adata),
            q_cell_type=get_q_type(self, cell_gene_sampler, key='qz_wc'),
            ouvain_cell_type=get_louvain_type(self, adata, args),
            leiden_cell_type=get_leiden_type(self, adata, args)
        )
        return result

    def get_loss(self, fwd_dict, data_dict, hyper_param_dict):
        pcz_w, qz_wc = fwd_dict['pcz_w'], fwd_dict['qz_wc']
        loss = -(qz_wc * (pcz_w + 1e-30).log()).sum(1).mean()
        
        new_items = {}

        if hyper_param_dict['epsilon']:
            BCE_gene_LINE = self.get_LINE_loss(
                fwd_dict, data_dict, hyper_param_dict)
            loss = loss + hyper_param_dict['epsilon'] * BCE_gene_LINE
            new_items['Llg'] = BCE_gene_LINE

        loss, new_items = get_gg_cc_loss(
            self, fwd_dict, data_dict, hyper_param_dict, loss, new_items)
        return loss, new_items


class LINE(nn.Module):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.emb_dim = args.emb_dim
        self.n_labels = args.n_labels
        self.device = device
        self.n_cells = adata.n_obs
        self.n_genes = adata.n_vars
        self.neg_samples = args.neg_samples
        self.batch_removal = args.max_lambda > 0

        self.w_cell_emb = get_emb(
            self.n_cells, self.emb_dim, 1).to(device)
        if self.batch_removal:
            self.batch_clf = nn.Sequential(
                nn.Linear(self.emb_dim, self.emb_dim // 2),
                nn.ReLU(),
                nn.Linear(self.emb_dim // 2, adata.obs.batch_indices.nunique())
            ).to(device)

            class RevGrad(Function):
                @staticmethod
                def forward(ctx, input_):
                    ctx.save_for_backward(input_)
                    output = input_
                    return output

                @staticmethod
                def backward(ctx, grad_output):
                    grad_input = None
                    if ctx.needs_input_grad[0]:
                        grad_input = -grad_output * args.revgrad_weight
                    return grad_input

            self.revgrad = RevGrad.apply
            
        if self.neg_samples:
            self.c_gene_emb = nn.Embedding(
                self.n_genes, self.emb_dim).to(device)
        else:
            self.gene_decoder = get_dec(
                self.n_genes, self.emb_dim, 1).to(device)
        self.init_emb()


    @staticmethod
    def _init_emb(emb):
        if isinstance(emb, nn.Linear):
            nn.init.xavier_uniform_(emb.weight.data)
            if emb.bias is not None:
                emb.bias.data.fill_(0.0)
        elif isinstance(emb, nn.Sequential):
            for child in emb:
                LINE._init_emb(child)

    def init_emb(self):
        for m in self.modules():
            self._init_emb(m)

    def get_cell_emb_weights(self):
        return {'w_cell_emb': self.w_cell_emb.weight.detach().cpu().numpy()}

    def forward(self, data_dict, hyper_param_dict):
        cells = data_dict['cells']
        w_cell_emb = self.w_cell_emb(cells)
        fwd_dict = dict(w_cell_emb=w_cell_emb)
        if self.batch_removal:
            batch_logit = self.batch_clf(self.revgrad(w_cell_emb))
            fwd_dict['batch_logit'] = batch_logit
        return fwd_dict

    def _get_c2g_loss_neg(self, fwd_dict, data_dict, hyper_param_dict,
                          emb_name):
        c_gene_emb = self.c_gene_emb(data_dict['genes'])
        c_gene_emb_neg = self.c_gene_emb(data_dict['neg_genes']) * (-1)
        w_cell_emb = fwd_dict[emb_name]
        pos = F.logsigmoid((c_gene_emb * w_cell_emb).sum(dim=-1))
        neg = F.logsigmoid(
            (c_gene_emb_neg * w_cell_emb.unsqueeze(1)).sum(dim=-1)
        ).sum(dim=-1)

        return (pos + hyper_param_dict['neg_weight'] * neg).mean() * (-1.)

    def get_LINE_loss(self, fwd_dict, data_dict, hyper_param_dict):
        if self.neg_samples:
            return self._get_c2g_loss_neg(
                fwd_dict, data_dict, hyper_param_dict, 'w_cell_emb')
        else:
            genes = data_dict['genes']
            recon_c_gene_LINE = self.gene_decoder(fwd_dict['w_cell_emb'])
            BCE_gene_LINE = F.cross_entropy(
                recon_c_gene_LINE, genes, reduction='sum') / genes.shape[0]
            return BCE_gene_LINE

    def get_loss(self, fwd_dict, data_dict, hyper_param_dict):
        LINE_loss = self.get_LINE_loss(fwd_dict, data_dict, hyper_param_dict)
        if hyper_param_dict['lambda']:
            batch_CE = F.cross_entropy(
                fwd_dict['batch_logit'],
                data_dict['batch_indices'],
                reduction='sum'
            ) / data_dict['cells'].shape[0]
            loss = hyper_param_dict['lambda'] * batch_CE + LINE_loss
            return loss, {'LINE': LINE_loss,'batchCE': batch_CE}
        else:
            return LINE_loss, {}

    def get_cell_type(self, cell_gene_sampler, adata=None, args=None, hard_accum=False):
        self.eval()
        louvain_cell_type = get_louvain_type(self, adata, args)
        k_cell_type = get_k_type(self)
        return {
            'louvain_cell_type': louvain_cell_type,
            'k_cell_type': k_cell_type
        }


class vGraphWithCellProfile(nn.Module):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        args.neg_samples = 0

        self.emb_dim = args.emb_dim
        self.n_labels = args.n_labels
        self.device = device
        self.n_cells = adata.n_obs
        self.n_genes = adata.n_vars
        self.n_batches = adata.obs.batch_indices.nunique()
        self.neg_samples = args.neg_samples
        self.batch_size = args.batch_size
        self.eval_batches = args.eval_batches
        self.encoder_depth = args.encoder_depth
        self.decoder_depth = args.decoder_depth
        self.batch_removal = args.max_lambda > 0
        self.cell_batch_scaling = args.cell_batch_scaling
        self.X = torch.FloatTensor(adata.X).to(self.device)
        # self.library_size = adata.X.sum(1)

        self.q_theta = nn.Sequential(
            nn.Linear(self.n_genes, self.emb_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.emb_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.emb_dim),
            nn.Dropout(0.1)
        ).to(device)
        self.mu_q_theta = nn.Linear(self.emb_dim, self.n_labels, bias=True).to(device)

        self.mu_decoder = nn.Parameter(torch.randn(self.n_labels, self.n_genes, device=device))
        self.logdisp_decoder = nn.Parameter(torch.randn(self.n_labels, self.n_genes, device=device))
        self.zi_decoder = nn.Parameter(torch.randn(self.n_labels, self.n_genes, device=device))

        self.E_result = None
            
        self.init_emb()

    @staticmethod
    def _init_emb(emb):
        if isinstance(emb, nn.Linear):
            nn.init.xavier_uniform_(emb.weight.data)
            if emb.bias is not None:
                emb.bias.data.fill_(0.0)
        elif isinstance(emb, nn.Sequential):
            for child in emb:
                vGraphWithCellProfile._init_emb(child)

    def init_emb(self):
        for m in self.modules():
            self._init_emb(m)

    def get_cell_emb_weights(self):
        w_emb = self.q_theta(self.X).detach().cpu().numpy()
        return {'w_cell_emb': w_emb}

    def _get_batch_indices_oh(self, data_dict):
        if 'batch_indices_oh' in data_dict:
            w_batch_id = data_dict['batch_indices_oh']
        else:
            batch_indices = data_dict['batch_indices'].unsqueeze(1)
            w_batch_id = torch.zeros((batch_indices.shape[0], self.n_batches), dtype=torch.float32, device=self.device)
            w_batch_id.scatter_(1, batch_indices, 1.)
            data_dict['batch_indices_oh'] = w_batch_id
        return w_batch_id

    def E_step(self, data_dict=dict(), hyper_param_dict=dict(E=False)):
        cells = data_dict['cells']
        q_theta = self.q_theta(cells)
        mu_q_theta = self.mu_q_theta(q_theta)
        pz_x = F.softmax(mu_q_theta, dim=-1)
        self.E_result = dict(pz_x=pz_x, qz=pz_x.detach())
        return self.E_result

    def forward(self, data_dict, hyper_param_dict=dict(E=False)):
        if hyper_param_dict['E']:
            self.E_step(data_dict, hyper_param_dict)
        fwd_dict = self.E_result

        cells, library_size = data_dict['cells'], data_dict['library_size']

        mu_x = F.softmax(self.mu_decoder, dim=-1).unsqueeze(0) * library_size.reshape(library_size.shape[0], 1, 1)
        disp_x = self.logdisp_decoder.exp().unsqueeze(0)
        zi_logits = self.zi_decoder.unsqueeze(0)

        # [batch_size, n_labels]
        log_pz_x = (self.E_result['pz_x'] + 1e-30).log()
        log_px_z = ZeroInflatedNegativeBinomial(mu=mu_x, theta=disp_x, zi_logits=zi_logits).log_prob(cells.unsqueeze(1)).sum(-1)
        log_pxz = log_pz_x + log_px_z
        fwd_dict['log_pxz'] = log_pxz
        return fwd_dict

    def get_cell_type(self, cell_gene_sampler, adata=None, args=None, hard_accum=False):
        self.eval()
        result = dict(
            q_cell_type=self.E_step(dict(cells=self.X))['qz'].argmax(1).detach().cpu().numpy(),
            ouvain_cell_type=get_louvain_type(self, adata, args),
            leiden_cell_type=get_leiden_type(self, adata, args)
        )
        return result

    def get_loss(self, fwd_dict, data_dict, hyper_param_dict):
        log_pxz, pz_x = fwd_dict['log_pxz'], fwd_dict['pz_x']

        loss = -(pz_x * log_pxz).sum(1).mean()
        new_items = {}

        if hyper_param_dict['epsilon']:
            BCE_gene_LINE = self.get_LINE_loss(
                fwd_dict, data_dict, hyper_param_dict)
            loss = loss + hyper_param_dict['epsilon'] * BCE_gene_LINE
            new_items['Llg'] = BCE_gene_LINE

        loss, new_items = get_gg_cc_loss(
            self, fwd_dict, data_dict, hyper_param_dict, loss, new_items)
        return loss, new_items

class vGraphEMCell(nn.Module):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        args.neg_samples = 0

        self.emb_dim = args.emb_dim
        self.n_labels = args.n_labels
        self.device = device
        self.n_cells = adata.n_obs
        self.n_genes = adata.n_vars
        self.n_batches = adata.obs.batch_indices.nunique()
        self.neg_samples = args.neg_samples
        self.batch_size = args.batch_size
        self.eval_batches = args.eval_batches
        self.encoder_depth = args.encoder_depth
        self.decoder_depth = args.decoder_depth
        self.batch_removal = args.max_lambda > 0
        self.norm_cells = args.norm_cells
        self.cell_batch_scaling = args.cell_batch_scaling
        self.library_size = adata.X.sum(1, keepdims=True)
        self.X = (adata.X / self.library_size) if self.norm_cells else adata.X
        self.batch_indices = adata.obs.batch_indices.astype(int)

        self.q_delta = nn.Sequential(
            nn.Linear(self.n_genes, self.emb_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.emb_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.emb_dim),
            nn.Dropout(0.1)
        ).to(device)
        self.mu_q_delta = nn.Linear(self.emb_dim, self.n_labels, bias=True).to(device)
        self.logsigma_q_delta = nn.Linear(self.emb_dim, self.n_labels, bias=True).to(device)


        # self.mu_decoder = nn.Parameter(torch.randn(self.n_labels, self.n_genes, device=device))
        # self.logdisp_decoder = nn.Parameter(torch.randn(self.n_labels, self.n_genes, device=device))
        # self.zi_decoder = nn.Parameter(torch.randn(self.n_labels, self.n_genes, device=device))
        self.gene_decoder = nn.Parameter(torch.randn(self.n_labels, self.n_genes, device=device))

        self.E_result = None
            
        self.init_emb()

    @staticmethod
    def _init_emb(emb):
        if isinstance(emb, nn.Linear):
            nn.init.xavier_uniform_(emb.weight.data)
            if emb.bias is not None:
                emb.bias.data.fill_(0.0)
        elif isinstance(emb, nn.Sequential):
            for child in emb:
                vGraphEMCell._init_emb(child)

    def init_emb(self):
        for m in self.modules():
            self._init_emb(m)

    def get_cell_emb_weights(self):
        return BlackHole()

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

        q_delta = self.q_delta(cells)
        mu_q_delta = self.mu_q_delta(q_delta)
        logsigma_q_delta = self.logsigma_q_delta(q_delta)

        # mu_x = F.softmax(self.mu_decoder, dim=-1).unsqueeze(0) * library_size.reshape(library_size.shape[0], 1, 1)
        # disp_x = self.logdisp_decoder.exp().unsqueeze(0)
        # zi_logits = self.zi_decoder.unsqueeze(0)
        x = F.log_softmax(self.gene_decoder, dim=-1).unsqueeze(0)
        log_pxz = (x * ((cells * library_size) if self.norm_cells else cells).unsqueeze(1)).sum(-1)

        # [batch_size, n_labels]
        # log_pxz = ZeroInflatedNegativeBinomial(mu=mu_x, theta=disp_x, zi_logits=zi_logits).log_prob(cells.unsqueeze(1)).sum(-1)
        if hyper_param_dict['E']:
            qz = F.softmax(log_pxz, dim=-1).detach()
            self.E_result = qz
        else:
            qz = self.E_result
        fwd_dict = dict(log_pxz=log_pxz, qz=qz)
        return fwd_dict

    def get_cell_type(self, cell_gene_sampler, adata=None, args=None, hard_accum=False):
        self.eval()
        if self.n_cells > self.batch_size:
            q_cell_types = []
            for start in range(0, self.n_cells, self.batch_size):
                cells = torch.FloatTensor(self.X[start: start + self.batch_size]).to(self.device)
                library_size = torch.FloatTensor(self.library_size[start: start + self.batch_size]).to(self.device)
                data_dict = dict(cells=cells, library_size=library_size,
                    cell_indices=torch.arange(start, min(start + self.batch_size, self.n_cells), device=self.device))
                if self.batch_removal or self.cell_batch_scaling:
                    batch_indices = torch.LongTensor(self.batch_indices[start: start + self.batch_size]).to(self.device)
                    data_dict['batch_indices'] = batch_indices
                q_cell_types.append(self(data_dict)['qz'].argmax(-1).detach().cpu())
            q_cell_type = torch.cat(q_cell_types, dim=0).numpy()
        else:
            cells = torch.FloatTensor(self.X).to(self.device)
            library_size = torch.FloatTensor(self.library_size).to(self.device)
            data_dict = dict(cells=cells, library_size=library_size, cell_indices=torch.arange(self.n_cells, device=self.device))
            if self.batch_removal or self.cell_batch_scaling:
                batch_indices = torch.LongTensor(self.batch_indices).to(self.device)
                data_dict['batch_indices'] = batch_indices
            q_cell_type = self(data_dict)['qz'].argmax(-1).detach().cpu().numpy()

        result = dict(
            q_cell_type=q_cell_type,
            # ouvain_cell_type=get_louvain_type(self, adata, args),
            # leiden_cell_type=get_leiden_type(self, adata, args)
        )
        return result

    def get_loss(self, fwd_dict, data_dict, hyper_param_dict):
        log_pxz, qz = fwd_dict['log_pxz'], fwd_dict['qz']

        loss = -(qz * log_pxz).sum(1).mean()
        new_items = {}

        if hyper_param_dict['epsilon']:
            BCE_gene_LINE = self.get_LINE_loss(
                fwd_dict, data_dict, hyper_param_dict)
            loss = loss + hyper_param_dict['epsilon'] * BCE_gene_LINE
            new_items['Llg'] = BCE_gene_LINE

        loss, new_items = get_gg_cc_loss(
            self, fwd_dict, data_dict, hyper_param_dict, loss, new_items)
        return loss, new_items


class scETM(nn.Module):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        args.neg_samples = 0

        self.emb_dim = args.emb_dim
        self.n_labels = args.n_labels
        self.n_topics = args.n_topics 
        self.device = device
        self.n_cells = adata.n_obs
        self.n_genes = adata.n_vars
        self.n_batches = adata.obs.batch_indices.nunique()
        self.batch_size = args.batch_size
        self.neg_samples = args.neg_samples
        self.eval_batches = args.eval_batches
        self.encoder_depth = args.encoder_depth
        self.decoder_depth = args.decoder_depth
        self.batch_removal = args.max_lambda > 0
        self.norm_cells = args.norm_cells
        self.cell_batch_scaling = args.cell_batch_scaling
        self.library_size = adata.X.sum(1, keepdims=True)
        self.X = (adata.X / self.library_size) if self.norm_cells else adata.X
        self.batch_indices = adata.obs.batch_indices.astype(int)
 
        self.q_delta = nn.Sequential(
            nn.Linear(self.n_genes + ((self.n_batches - 1) if self.batch_removal else 0), self.emb_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.emb_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.emb_dim),
            nn.Dropout(0.1)
        ).to(device)
        self.mu_q_delta = nn.Linear(self.emb_dim, self.n_topics, bias=True).to(device)
        self.logsigma_q_delta = nn.Linear(self.emb_dim, self.n_topics, bias=True).to(device)

        self.beta = nn.Linear(self.n_topics + ((self.n_batches - 1) if self.batch_removal else 0), self.n_genes, bias=True).to(device)
        # self.beta = nn.Parameter(torch.randn(self.n_topics, self.n_genes, device=device))
        if self.cell_batch_scaling:
            self.cell_bias = nn.Parameter(torch.randn(self.n_cells, 1)).to(device)
            self.gene_bias = nn.Parameter(torch.randn(self.n_batches, self.n_genes)).to(device)

        ## cap log variance within [-10, 10]
        self.max_logsigma = 10
        self.min_logsigma = -10

        self.init_emb()

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
        if self.n_cells > self.batch_size:
            w_embs = []
            for start in range(0, self.n_cells, self.batch_size):
                cells = torch.FloatTensor(self.X[start: start + self.batch_size]).to(self.device)
                library_size = torch.FloatTensor(self.library_size[start: start + self.batch_size]).to(self.device)
                data_dict = dict(cells=cells, library_size=library_size,
                    cell_indices=torch.arange(start, min(start + self.batch_size, self.n_cells), device=self.device))
                if self.batch_removal or self.cell_batch_scaling:
                    batch_indices = torch.LongTensor(self.batch_indices[start: start + self.batch_size]).to(self.device)
                    data_dict['batch_indices'] = batch_indices
                w_embs.append(self(data_dict)['theta'].detach().cpu())
            w_emb = torch.cat(w_embs, dim=0).numpy()
        else:
            cells = torch.FloatTensor(self.X).to(self.device)
            library_size = torch.FloatTensor(self.library_size).to(self.device)
            data_dict = dict(cells=cells, library_size=library_size, cell_indices=torch.arange(self.n_cells, device=self.device))
            if self.batch_removal or self.cell_batch_scaling:
                batch_indices = torch.LongTensor(self.batch_indices).to(self.device)
                data_dict['batch_indices'] = batch_indices
            w_emb = self(data_dict)['theta'].detach().cpu().numpy()
        return {'w_cell_emb': w_emb}

    def forward(self, data_dict, hyper_param_dict=dict(E=True)):
        cells, library_size = data_dict['cells'], data_dict['library_size']
        batch_size = cells.shape[0]
        
        if self.batch_removal:
            batch_indices_oh = self._get_batch_indices_oh(data_dict)
            q_delta = self.q_delta(torch.cat((cells, batch_indices_oh), dim=1))
        else:
            q_delta = self.q_delta(cells)
        mu_q_delta = self.mu_q_delta(q_delta)
        logsigma_q_delta = self.logsigma_q_delta(q_delta).clamp_(self.min_logsigma, self.max_logsigma)
        p_delta = Independent(Normal(
            loc=torch.zeros(batch_size, self.n_topics, dtype=torch.float32, device=self.device),
            scale=torch.ones(batch_size, self.n_topics, dtype=torch.float32, device=self.device)
        ), 1)
        q_delta = Independent(Normal(
            loc=mu_q_delta,
            scale=logsigma_q_delta.exp()
        ), 1)
        delta = q_delta.rsample()
        theta = F.softmax(delta, dim=-1)  # [batch_size, n_topics]

        if self.batch_removal:
            recon_logit = self.beta(torch.cat((theta, batch_indices_oh), dim=1))
        else:
            recon_logit = self.beta(theta)  # [batch_size, n_genes]
        if self.cell_batch_scaling:
            recon_logit += self.cell_bias[data_dict['cell_indices']] * self.gene_bias[data_dict['batch_indices']]
        # recon_logit = torch.mm(theta, F.softmax(self.beta, dim=-1))
        fwd_dict = dict(
            nll=(-F.log_softmax(recon_logit, dim=-1) * ((cells) if self.norm_cells else cells)).sum(-1).mean(),
            # nll=(-recon_logit.log() * cells).sum(-1).mean(),
            kl=(q_delta.log_prob(delta) - p_delta.log_prob(delta)).mean(),
            theta=theta
        )
        return fwd_dict

    def get_cell_type(self, cell_gene_sampler, adata=None, args=None, hard_accum=False):
        self.eval()

        theta = self.get_cell_emb_weights()['w_cell_emb']
        p_cell_type = theta.argmax(-1)

        result = dict(
            p_cell_type=p_cell_type,
            ouvain_cell_type=get_louvain_type(self, adata, args),
            leiden_cell_type=get_leiden_type(self, adata, args)
        )
        return result

    def get_loss(self, fwd_dict, data_dict, hyper_param_dict):
        nll, kl = fwd_dict['nll'], fwd_dict['kl']
        loss = nll + hyper_param_dict['beta'] * kl
        new_items = dict(nll=nll, kl=kl)

        if hyper_param_dict['epsilon']:
            BCE_gene_LINE = self.get_LINE_loss(
                fwd_dict, data_dict, hyper_param_dict)
            loss = loss + hyper_param_dict['epsilon'] * BCE_gene_LINE
            new_items['Llg'] = BCE_gene_LINE

        loss, new_items = get_gg_cc_loss(
            self, fwd_dict, data_dict, hyper_param_dict, loss, new_items)
        return loss, new_items


class NewModel(nn.Module):
    def __init__(self, adata: anndata.AnnData, args,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        args.neg_samples = 0

        self.emb_dim = args.emb_dim
        self.n_labels = args.n_labels
        self.n_topics = args.n_topics 
        self.device = device
        self.n_cells = adata.n_obs
        self.n_genes = adata.n_vars
        self.n_batches = adata.obs.batch_indices.nunique()
        self.batch_size = args.batch_size
        self.neg_samples = args.neg_samples
        self.eval_batches = args.eval_batches
        self.encoder_depth = args.encoder_depth
        self.decoder_depth = args.decoder_depth
        self.batch_removal = args.max_lambda > 0
        self.cell_batch_scaling = args.cell_batch_scaling
        self.norm_cells = args.norm_cells
        self.library_size = adata.X.sum(1, keepdims=True)
        self.X = (adata.X / self.library_size) if self.norm_cells else adata.X
        self.batch_indices = adata.obs.batch_indices.astype(int)

        self.E_result = None
 
        self.q_delta = nn.Sequential(
            nn.Linear(self.n_genes + ((self.n_batches - 1) if self.batch_removal else 0), self.emb_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.emb_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.emb_dim),
            nn.Dropout(0.1)
        ).to(device)
        self.mu_q_delta = nn.Linear(self.emb_dim, self.n_topics, bias=True).to(device)
        self.logsigma_q_delta = nn.Linear(self.emb_dim, self.n_topics, bias=True).to(device)

        # self.beta = nn.Linear(self.n_topics + ((self.n_batches - 1) if self.batch_removal else 0), self.n_genes, bias=True).to(device)
        self.beta = nn.Parameter(torch.randn(self.n_genes, self.n_topics, device=device))

        ## cap log variance within [-10, 10]
        self.max_logsigma = 10
        self.min_logsigma = -10

        self.init_emb()

    @staticmethod
    def _init_emb(emb):
        if isinstance(emb, nn.Linear):
            nn.init.xavier_uniform_(emb.weight.data)
            if emb.bias is not None:
                emb.bias.data.fill_(0.0)
        elif isinstance(emb, nn.Sequential):
            for child in emb:
                NewModel._init_emb(child)

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
        if self.n_cells > self.batch_size:
            w_embs = []
            for start in range(0, self.n_cells, self.batch_size):
                cells = torch.FloatTensor(self.X[start: start + self.batch_size]).to(self.device)
                library_size = torch.FloatTensor(self.library_size[start: start + self.batch_size]).to(self.device)
                data_dict = dict(cells=cells, library_size=library_size,
                    cell_indices=torch.arange(start, min(start + self.batch_size, self.n_cells), device=self.device))
                if self.batch_removal or self.cell_batch_scaling:
                    batch_indices = torch.LongTensor(self.batch_indices[start: start + self.batch_size]).to(self.device)
                    data_dict['batch_indices'] = batch_indices
                w_embs.append(self(data_dict)['theta'].detach().cpu())
            w_emb = torch.cat(w_embs, dim=0).numpy()
        else:
            cells = torch.FloatTensor(self.X).to(self.device)
            library_size = torch.FloatTensor(self.library_size).to(self.device)
            data_dict = dict(cells=cells, library_size=library_size, cell_indices=torch.arange(self.n_cells, device=self.device))
            if self.batch_removal or self.cell_batch_scaling:
                batch_indices = torch.LongTensor(self.batch_indices).to(self.device)
                data_dict['batch_indices'] = batch_indices
            w_emb = self(data_dict)['theta'].detach().cpu().numpy()
        return {'w_cell_emb': w_emb}

    def forward(self, data_dict, hyper_param_dict=dict(E=False)):
        cells, library_size = data_dict['cells'], data_dict['library_size']
        batch_size = cells.shape[0]
        
        if self.batch_removal:
            batch_indices_oh = self._get_batch_indices_oh(data_dict)
            q_delta = self.q_delta(torch.cat((cells, batch_indices_oh), dim=1))
        else:
            q_delta = self.q_delta(cells)
        mu_q_delta = self.mu_q_delta(q_delta)
        logsigma_q_delta = self.logsigma_q_delta(q_delta).clamp_(self.min_logsigma, self.max_logsigma)
        p_delta = Independent(Normal(
            loc=torch.zeros(batch_size, self.n_topics, dtype=torch.float32, device=self.device),
            scale=torch.ones(batch_size, self.n_topics, dtype=torch.float32, device=self.device)
        ), 1)
        q_delta = Independent(Normal(
            loc=mu_q_delta,
            scale=logsigma_q_delta.exp()
        ), 1)
        delta = q_delta.rsample()
        theta = F.softmax(delta, dim=-1)  # [batch_size, n_topics]

        # [n_genes, n_topics]
        log_beta = F.log_softmax(self.beta, dim=0)
        # [batch_size, 1, n_topics]
        log_theta = F.log_softmax(delta, dim=-1).unsqueeze(1)
        # [batch_size, n_genes, n_topics]
        log_pz_d = log_theta
        log_px_z = log_beta.unsqueeze(0) * ((cells * library_size) if self.norm_cells else cells).unsqueeze(2)
        log_pzx_d = log_pz_d + log_px_z
        if hyper_param_dict['E']:
            pz_dx = F.softmax(log_pzx_d, dim=-1)
            self.E_result = pz_dx.detach()
        else:
            pz_dx = self.E_result

        fwd_dict = dict(
            nll=-(pz_dx * log_pzx_d).sum(-1).sum(-1).mean(),
            kl_delta=(q_delta.log_prob(delta) - p_delta.log_prob(delta)).mean(),
            theta=theta 
        )
        return fwd_dict

    def get_cell_type(self, cell_gene_sampler, adata=None, args=None, hard_accum=False):
        self.eval()

        theta = self.get_cell_emb_weights()['w_cell_emb']
        p_cell_type = theta.argmax(-1)

        result = dict(
            p_cell_type=p_cell_type,
            ouvain_cell_type=get_louvain_type(self, adata, args),
            leiden_cell_type=get_leiden_type(self, adata, args)
        )
        return result

    def get_loss(self, fwd_dict, data_dict, hyper_param_dict):
        # nll, kl_delta, kl_z = fwd_dict['nll'], fwd_dict['kl_delta'], fwd_dict['kl_z']
        # loss = nll + hyper_param_dict['beta'] * kl_delta + hyper_param_dict['eta'] * kl_z
        # new_items = dict(nll=nll, kl_delta=kl_delta, kl_z=kl_z)
        nll, kl_delta = fwd_dict['nll'], fwd_dict['kl_delta']
        loss = nll + hyper_param_dict['beta'] * kl_delta
        new_items = dict(nll=nll, kl_delta=kl_delta)

        if hyper_param_dict['epsilon']:
            BCE_gene_LINE = self.get_LINE_loss(
                fwd_dict, data_dict, hyper_param_dict)
            loss = loss + hyper_param_dict['epsilon'] * BCE_gene_LINE
            new_items['Llg'] = BCE_gene_LINE

        loss, new_items = get_gg_cc_loss(
            self, fwd_dict, data_dict, hyper_param_dict, loss, new_items)
        return loss, new_items


def get_gg_cc_loss(self, fwd_dict: dict, data_dict: dict,
                   hyper_param_dict: dict, loss, new_items: dict):
    if hyper_param_dict['gamma']:
        g1, g2 = data_dict['g1'], data_dict['g2']
        g1_emb = self.gene_emb(g1)
        recon_g2 = self.gene_decoder(g1_emb)
        g2_loss = F.cross_entropy(recon_g2, g2, reduction='mean')
        g2_emb = self.gene_emb(g2)
        recon_g1 = self.gene_decoder(g2_emb)
        g1_loss = F.cross_entropy(recon_g1, g1, reduction='mean')
        gg_loss = g1_loss + g2_loss

        loss = loss + hyper_param_dict['gamma'] * gg_loss
        new_items['gg'] = gg_loss

    # cell2cell reconstruction loss
    if hyper_param_dict['delta']:
        c1, c2 = data_dict['c1'], data_dict['c2']
        c1_emb = self.cell_emb(c1)
        recon_c2 = self.cell_decoder(c1_emb)
        c2_loss = F.cross_entropy(recon_c2, c2, reduction='mean')
        c2_emb = self.cell_emb(c2)
        recon_c1 = self.cell_decoder(c2_emb)
        c1_loss = F.cross_entropy(recon_c1, c1, reduction='mean')
        cc_loss = c1_loss + c2_loss

        loss = loss + hyper_param_dict['delta'] * cc_loss
        new_items['cc'] = cc_loss

    return loss, new_items


def get_p_type(model, adata=None, key='pz'):
    all_cells = torch.arange(
        0, model.n_cells, 1, dtype=torch.long, device=model.device)
    genes = torch.zeros_like(all_cells, device=model.device)
    data_dict = dict(cells=all_cells, genes=genes)

    if adata is not None:
        data_dict['batch_indices'] = torch.LongTensor(adata.obs.batch_indices.astype(int).values).to(model.device)

    prior_cell_type = model(data_dict)[key].argmax(axis=-1)
    prior_cell_type = prior_cell_type.cpu().data.numpy()
    return prior_cell_type


def get_q_type(model, cell_gene_sampler, hard_accum=False, key='qz'):
    posterior_cell_type = torch.zeros(
        (model.n_cells, model.n_labels), device=model.device)

    for start in range(model.eval_batches):
        print('Testing: {:7d}/{:7d}{:33s}'.format(
            start, model.eval_batches, ''), end='\r')
        data_dict = cell_gene_sampler.pipeline.get_message()
        q = model(data_dict)[key].detach()

        if hard_accum:
            q_argmax = q.argmax(dim=-1)
            idx = torch.stack([data_dict['cells'], q_argmax])
            val = torch.ones([q_argmax.shape[0]], device=model.device)
        else:
            axis_x = data_dict['cells'].unsqueeze(1).expand_as(q).flatten()
            axis_y = torch.arange(q.shape[1], dtype=torch.long, device=model.device).unsqueeze(
                0).expand_as(q).flatten()
            idx = torch.stack([axis_x, axis_y])
            val = q.flatten()
        posterior_cell_type += torch.sparse.FloatTensor(
            idx, val, posterior_cell_type.shape).to_dense()

    posterior_cell_type = posterior_cell_type.argmax(dim=-1)
    posterior_cell_type = posterior_cell_type.cpu().data.numpy()
    return posterior_cell_type


def get_louvain_type(model, adata, args):
    print('Performing louvain clustering...', flush=True, end='\r')
    adata.obsm['w_cell_emb'] = model.get_cell_emb_weights()['w_cell_emb']
    for _ in range(3):
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep='w_cell_emb')
        sc.tl.louvain(adata, resolution=args.louvain_resolution)
        if args.fix_resolution:
            break
        if adata.obs.louvain.nunique() < args.n_labels / 2:
            args.louvain_resolution *= 2
        elif adata.obs.louvain.nunique() < args.n_labels / 1.1:
            args.louvain_resolution *= 1.2
        elif adata.obs.louvain.nunique() > args.n_labels * 2:
            args.louvain_resolution /= 2
        elif adata.obs.louvain.nunique() > args.n_labels * 1.1:
            args.louvain_resolution /= 1.2
        else:
            break
    return adata.obs['louvain'].astype(int)


def get_leiden_type(model, adata, args):
    print('Performing leiden clustering...', flush=True, end='\r')
    adata.obsm['w_cell_emb'] = model.get_cell_emb_weights()['w_cell_emb']
    for _ in range(3):
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep='w_cell_emb')
        sc.tl.leiden(adata, resolution=args.leiden_resolution)
        if args.fix_resolution:
            break
        if adata.obs.leiden.nunique() < args.n_labels / 2:
            args.leiden_resolution *= 2
        elif adata.obs.leiden.nunique() < args.n_labels / 1.1:
            args.leiden_resolution *= 1.2
        elif adata.obs.leiden.nunique() > args.n_labels * 2:
            args.leiden_resolution /= 2
        elif adata.obs.leiden.nunique() > args.n_labels * 1.1:
            args.leiden_resolution /= 1.2
        else:
            break
    return adata.obs['leiden'].astype(int)


def get_k_type(model):
    print('Performing k-means clustering...', flush=True, end='\r')
    kmeans = KMeans(n_clusters=model.n_labels, n_init=20)
    arr = model.get_cell_emb_weights()['w_cell_emb']
    kmeans.fit_transform(arr)
    kmeans_cell_type = kmeans.labels_
    return kmeans_cell_type

class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __getattr__(self, name):
        def method(*args, **kwargs):
            pass
        return method