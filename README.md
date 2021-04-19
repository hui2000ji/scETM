# scETM: single-cell Embedded Topic Model
A generative topic model that facilitates integrative analysis of large-scale single-cell RNA sequencing data.

The full description of scETM and its application on published single cell RNA-seq datasets are available [here](https://www.biorxiv.org/content/10.1101/2021.01.13.426593v1).

This repository includes detailed instructions for installation and requirements, demos, and scripts used for the benchmarking of 7 other state-of-art methods.


## Contents ##

- [scETM: single-cell Embedded Topic Model](#scetm-single-cell-embedded-topic-model)
  - [Contents](#contents)
  - [1 Model Overview](#1-model-overview)
  - [2 Installation](#2-installation)
  - [3 Usage](#3-usage)
    - [Required data](#required-data)
    - [p-scETM](#p-scetm)
  - [4 Benchmarking](#4-benchmarking)

## 1 Model Overview

![](doc/scETM.png "scETM model overview")
**(a)** Probabilistic graphical model of scETM. We model the scRNA-profile read count matrix y<sub>d,g</sub> in cell d and gene g across S subjects or studies by a multinomial distribution with the rate parameterized by cell topic mixture θ, topic embedding α, gene embedding ρ, and batch effects λ. **(b)** Matrix factorization view of scETM. **(c)** Encoder architecture for inferring the cell topic mixture θ.

## 2 Installation
Python version: 3.7+
scETM is included in PyPI, so you can install it by

```bash
pip install scETM
```

To enable GPU computing (which significantly boosts the performance), please install [PyTorch](https://pytorch.org/) with GPU support before installing scETM.

## 3 Usage
**A step-by-step scETM tutorial can be found in [here](/notebooks/scETM%20introductory%20tutorial.ipynb).**

### Required data
scETM requires a cells-by-genes matrix `adata` as input, in the format of an AnnData object. Detailed description about AnnData can be found [here](https://anndata.readthedocs.io/en/latest/).

By default, scETM looks for batch information in the 'batch_indices' column of the `adata.obs` DataFrame, and cell type identity in the 'cell_types' column. If your data stores the batch and cell type information in different columns, pass them to the `batch_col` and `cell_type_col` arguments, respectively, when calling scETM functions.

### p-scETM
p-scETM is a variant of scETM where part or all of the the gene embedding matrix ρ is fixed to a pathways-by-genes matrix, which can be downloaded from the [pathDIP4 pathway database](http://ophid.utoronto.ca/pathDIP/Download.jsp). We only keep pathways that contain more than 5 genes.

If it is desired to fix the gene embedding matrix ρ during training, let trainable_gene_emb_dim be zero. In this case, the gene set used to train the model would be the intersection of the genes in the scRNA-seq data and the genes in the gene-by-pathway matrix. Otherwise, if trainable_gene_emb_dim is set to a positive value, all the genes in the scRNA-seq data would be kept.

```python
from scETM import scETM, UnsupervisedTrainer, prepare_for_transfer
import anndata

# Prepare the source dataset, Mouse Pancreas
mp = anndata.read_h5ad("MousePancreas.h5ad")
# Initialize model
model = scETM(mp.n_vars, mp.obs.batch_indices.nunique(), enable_batch_bias=True)
# The trainer object will set up the random seed, optimizer, training and evaluation loop, checkpointing and logging.
trainer = UnsupervisedTrainer(model, mp, train_instance_name="MP", ckpt_dir="../results")
# Train the model on adata for 12000 epochs, and evaluate every 1000 epochs. Use 4 threads to sample minibatches.
trainer.train(n_epochs=12000, eval_every=1000, n_samplers=4)

# Load the target dataset, Human Pancreas
hp = anndata.read_h5ad('HumanPancreas.h5ad')
# Align the source dataset's gene names (which are mouse genes) to the target dataset (which are human genes)
mp_genes = mp.var_names.str.upper()
mp_genes.drop_duplicates(inplace=True)
# Generate a new model and a modified dataset from the previously trained model and the mp_genes
model, hp = prepare_for_transfer(model, hp, mp_genes,
	keep_tgt_unique_genes=True,    # Keep target-unique genes in the model and the target dataset
    fix_shared_genes=True          # Fix parameters related to shared genes in the model
)
# Instantiate another trainer to fine-tune the model
trainer = UnsupervisedTrainer(model, hp, train_instance_name="HP_all_fix", ckpt_dir="../results", init_lr=5e-4)
trainer.train(n_epochs=800, eval_every=200)
```


<a name="benchmarking"></a>
## 4 Benchmarking
The commands used for running [Harmony](https://github.com/immunogenomics/harmony), [Scanorama](https://github.com/brianhie/scanorama), [Seurat](https://satijalab.org/seurat/), [scVAE-GM](https://github.com/scvae/scvae), [scVI](https://github.com/YosefLab/scvi-tools), [LIGER](https://macoskolab.github.io/liger/), [scVI-LD](https://www.biorxiv.org/content/10.1101/737601v1.full.pdf) are available in the [scripts](/scripts) folder.
