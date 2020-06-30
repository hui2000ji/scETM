from typing import Union

import numpy as np
import anndata
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class ModelConfig:
    def __init__(self, n_enc_layers, n_dec_layers):
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers

    def get_emb(self, num_emb, emb_dim):
        if self.n_enc_layers == 1:
            return nn.Embedding(num_emb, emb_dim)
        enc_layers = [nn.Embedding(num_emb, emb_dim)]
        for _ in range(self.n_enc_layers - 1):
            enc_layers.append(nn.Linear(emb_dim, emb_dim))
            enc_layers.append(nn.ReLU())
        return nn.Sequential(*enc_layers)

    def get_dec(self, num_emb, emb_dim):
        if self.n_dec_layers == 1:
            return nn.Linear(emb_dim, num_emb)
        dec_layers = []
        for _ in range(self.n_dec_layers - 1):
            dec_layers.append(nn.Linear(emb_dim, emb_dim))
            dec_layers.append(nn.ReLU())
        dec_layers.append(nn.Linear(emb_dim, num_emb))
        return nn.Sequential(*dec_layers)


config = ModelConfig(1, 1)


class CellGeneModel(nn.Module):
    def __init__(self, adata: anndata.AnnData, args, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(CellGeneModel, self).__init__()
        self.emb_dim = args.emb_dim
        self.n_labels = args.n_labels
        self.device = device
        self.n_cells = adata.n_obs
        self.n_genes = adata.n_vars
        self.emb_combine = args.emb_combine
        self.decouple_pq = args.decouple_pq
        self.cell_type_dec = args.cell_type_dec
        self.neg_samples = args.neg_samples
        self.batch_size = args.batch_size
        self.eval_batches = args.eval_batches

        self.cell_type_emb = nn.Linear(self.emb_dim, self.n_labels, bias=False).to(device)
        if self.decouple_pq:
            if self.emb_combine == 'cat':
                self.cg_type_emb = nn.Linear(self.emb_dim * 2, self.n_labels).to(device)
            else:
                self.cg_type_emb = nn.Linear(self.emb_dim, self.n_labels).to(device)
        else:
            if self.emb_combine == 'cat':
                self.emb_compressor = nn.Linear(self.emb_dim * 2, self.emb_dim).to(device)
            self.cg_type_emb = self.cell_type_emb
        if self.cell_type_dec:
            self.cell_type_dec = nn.Linear(self.n_labels, self.emb_dim).to(device)
        self.cell_emb = config.get_emb(self.n_cells, self.emb_dim).to(device)
        self.gene_emb = config.get_emb(self.n_genes, self.emb_dim).to(device)

        if self.neg_samples:
            self.c_gene_emb = config.get_emb(self.n_genes, self.emb_dim).to(device)
        else:
            self.gene_decoder = config.get_dec(self.n_genes, self.emb_dim).to(device)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def get_cell_emb_weights(self):
        return {'w_cell_emb': self.cell_emb.weight.detach().cpu().numpy()}

    def forward(self, data_dict, hyper_param_dict=dict(tau=1.)):
        cells, genes = data_dict['cells'], data_dict['genes']
        tau = hyper_param_dict['tau']

        w_emb = self.cell_emb(cells).to(self.device)
        c_emb = self.gene_emb(genes).to(self.device)

        if self.emb_combine == 'prod':
            qz_logit = self.cg_type_emb(w_emb * c_emb)
        elif self.emb_combine == 'sum':
            qz_logit = self.cg_type_emb(w_emb + c_emb)
        elif self.emb_combine == 'cat':
            if self.decouple_pq:
                qz_logit = self.cg_type_emb(torch.cat([w_emb, c_emb], dim=1))
            else:
                qz_logit = self.cg_type_emb(self.emb_compressor(torch.cat([w_emb, c_emb], dim=1)))
        else:
            raise ValueError('emb_combine must be either prod, sum or cat')

        if self.training:
            qz = F.gumbel_softmax(logits=qz_logit, tau=tau, hard=True)
        else:
            tmp = qz_logit.argmax(dim=-1).reshape(qz_logit.shape[0], 1)
            qz = torch.zeros(qz_logit.shape).to(self.device).scatter_(1, tmp, 1.)

        pz_logit = self.cell_type_emb(w_emb)

        if self.cell_type_dec is False:
            recon_c_emb = torch.mm(qz, self.cell_type_emb.weight)
        else:
            recon_c_emb = self.cell_type_dec(qz)
        fwd_dict = {
            "qz": F.softmax(qz_logit, dim=-1),
            "pz": F.softmax(pz_logit, dim=-1),
            "w_emb": w_emb
        }
        if self.neg_samples:
            fwd_dict["recon_c_emb"] = recon_c_emb
        else:
            recon_c_logit = self.gene_decoder(recon_c_emb)
            fwd_dict["recon_c_gene"] = recon_c_logit
        return fwd_dict

    def get_cell_type(self, cell_gene_sampler, hard_accum=False):
        self.eval()

        all_cells = torch.arange(0, self.n_cells, 1, dtype=torch.long, device=self.device)
        genes = torch.zeros_like(all_cells, device=self.device)
        prior_cell_type = self(dict(cells=all_cells, genes=genes))['pz']
        prior_cell_type = prior_cell_type.cpu().data.numpy().argmax(axis=-1)

        posterior_cell_type = torch.zeros((self.n_cells, self.n_labels)).cpu()

        for start in range(self.eval_batches):
            print('Testing: {:7d}/{:7d}                                 '.format(start, self.eval_batches),
                  end='\r')
            data_dict = cell_gene_sampler.pipeline.get_message()
            q = self(data_dict)['qz']
            q = q[:self.batch_size].cpu()

            q_argmax = q.argmax(dim=-1).cpu()

            for idx, (w, c) in enumerate(zip(data_dict['cells'].cpu().numpy(), data_dict['genes'].cpu().numpy())):
                if hard_accum:
                    posterior_cell_type[w, q_argmax[idx]] += 1
                else:
                    posterior_cell_type[w, :] += q[idx, :]

        posterior_cell_type = posterior_cell_type.cpu().data.numpy()
        posterior_cell_type = posterior_cell_type.argmax(axis=-1)
        kmeans = KMeans(n_clusters=self.n_labels, n_init=20)
        arr = np.array(self.cell_emb.weight.detach().cpu().numpy())
        kmeans.fit_transform(arr)
        kmeans_cell_type = kmeans.labels_
        return {'p_cell_type': prior_cell_type,
                'q_cell_type': posterior_cell_type,
                'k_cell_type': kmeans_cell_type}

    def _get_c2g_loss(self, fwd_dict, data_dict, hyper_param_dict):
        genes = data_dict['genes']
        return F.cross_entropy(fwd_dict['recon_c_gene'], genes, reduction='sum') / genes.shape[0]

    def _get_c2g_loss_neg(self, fwd_dict, data_dict, hyper_param_dict, emb_name):
        c_gene_emb = self.c_gene_emb(data_dict['genes'])
        c_gene_emb_neg = self.c_gene_emb(data_dict['neg_genes']) * (-1)
        recon_c_gene_emb = fwd_dict[emb_name]
        pos = F.logsigmoid((c_gene_emb * recon_c_gene_emb).sum(dim=-1))
        neg = F.logsigmoid(
            (c_gene_emb_neg * recon_c_gene_emb.unsqueeze(1)).sum(dim=-1)
        ).sum(dim=-1)

        return (pos + hyper_param_dict['neg_weight'] * neg).mean() * (-1.)

    def get_c2g_loss(self, fwd_dict, data_dict, hyper_param_dict):
        if self.neg_samples:
            return self._get_c2g_loss_neg(fwd_dict, data_dict, hyper_param_dict, 'recon_c_emb')
        else:
            return self._get_c2g_loss(fwd_dict, data_dict, hyper_param_dict)

    def get_LINE_loss(self, fwd_dict, data_dict, hyper_param_dict):
        if self.neg_samples:
            return self._get_c2g_loss_neg(fwd_dict, data_dict, hyper_param_dict, 'w_emb')
        else:
            genes = data_dict['genes']
            recon_c_gene_LINE = self.gene_decoder(fwd_dict['w_emb'])
            BCE_gene_LINE = F.cross_entropy(recon_c_gene_LINE, genes, reduction='sum') / genes.shape[0]
            return BCE_gene_LINE

    def get_loss(self, fwd_dict, data_dict, hyper_param_dict):
        qz, pz = fwd_dict['qz'], fwd_dict['pz']
        cells = data_dict['cells']

        qz_plus = qz + 1e-20
        log_qz = torch.log(qz_plus)
        KLD = torch.sum(qz * (log_qz - torch.log(pz + 1e-20)), dim=-1).mean()

        BCE_gene = self.get_c2g_loss(fwd_dict, data_dict, hyper_param_dict)
        loss = BCE_gene + hyper_param_dict['beta'] * KLD
        new_items = {'KL': KLD, 'c2g': BCE_gene}

        if hyper_param_dict['eta']:
            cell_type_dist = None
            for i in range(self.n_labels):
                for j in range(self.n_labels):
                    if i == j:
                        continue
                    new_dist = ((self.cell_type_emb.weight[i] - self.cell_type_emb.weight[j])**2).sum()
                    if cell_type_dist is None:
                        cell_type_dist = new_dist
                    else:
                        cell_type_dist = cell_type_dist + new_dist
            cell_type_dist = cell_type_dist / (self.n_labels * (self.n_labels - 1))
            if cell_type_dist < 10:
                loss = loss - cell_type_dist * hyper_param_dict['eta']
            new_items['Dct'] = cell_type_dist


        if hyper_param_dict['epsilon']:
            BCE_gene_LINE = self.get_LINE_loss(fwd_dict, data_dict, hyper_param_dict)
            loss = loss + hyper_param_dict['epsilon'] * BCE_gene_LINE
            new_items['Llg'] = BCE_gene_LINE

        # word reconstruction loss from qz
        if hyper_param_dict['zeta']:
            recon_w_cell = torch.mm(qz, self.cell_type_emb.weight)
            BCEw_cell = F.cross_entropy(recon_w_cell, cells, reduction='sum') / cells.shape[0]
            BCEw = self.g2c_factor * BCEw_cell

            loss = loss + hyper_param_dict['zeta'] * BCEw
            new_items['Lwc'] = BCEw_cell

        loss, new_items = get_gg_cc_loss(self, fwd_dict, data_dict, hyper_param_dict, loss, new_items)
        return loss, new_items

def get_gg_cc_loss(self,
                   fwd_dict : dict, data_dict : dict, hyper_param_dict : dict, loss, new_items: dict):
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