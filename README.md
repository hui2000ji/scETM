# scETM: single-cell Embedded Topic Model
A generative topic model that facilitates integrative analysis of large-scale single-cell RNA sequencing data.

The full description of scETM and its application on published single cell RNA-seq datasets are available in ... (to be added once on bioarxiv).

This repository includes detailed instructions for installation and requirements, demos, and scripts used for the benchmarking of 7 other state-of-art methods.


## Contents ##

1. [Model Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
  - [Required data](#requireddata)
  - [Example usage](#usage)
4. [Benchmarking](#benchmarking)
  - [Data simulation](#simulation)
  - [Running baseline methods](#baseline)


<a name="overview"></a>
## 1 Model Overview

![](doc/scETM.png "scETM model overview")
**(a)** Probabilistic graphical model of scETM. We model the scRNA-profile read count matrix y<sub>d,g</sub> in cell d and gene g across S subjects or studies by a multinomial distribution with the rate parameterized by cell topic mixture θ, topic embedding α, gene embedding ρ, and batch effects λ. **(b)** Matrix factorization view of scETM. **(c)** Encoder architecture for inferring the cell topic mixture θ.

<a name="installation"></a>
## 2 Installation
- TODO: add the requirements.txt

<a name="usage"></a>
## 3 Usage

<a name="data"></a>
### Required data
scETM requires a cells-by-genes matrix as input, in the format of an AnnData object. Detailed description about AnnData can be found [here](https://anndata.readthedocs.io/en/latest/).

<a name="usage"></a>
### Example usage
1. scETM
```
$ python train.py \
 --model scETM \
 --norm-cells \
 --n-topics 100 \
 --h5ad-path data/MousePancreas.h5ad
```

2. pathway-informed scETM
  - to be added by Huiyu

<a name="benchmarking"></a>
## 4 Benchmarking

<a name="simulation"></a>
### Data simulation
- to be added by Huiyu

<a name="baseline"></a>
### Running baseline methods
The commands used for running [Harmony](https://github.com/immunogenomics/harmony), [Scanorama](https://github.com/brianhie/scanorama), [Seurat](https://satijalab.org/seurat/), [scVAE-GM](https://github.com/scvae/scvae), [scVI](https://github.com/YosefLab/scvi-tools), [LIGER](https://macoskolab.github.io/liger/), [scVI-LD](https://www.biorxiv.org/content/10.1101/737601v1.full.pdf) are available in the [baselines](/baselines) folder.

- TODO: scripts for harmony, scanorama, seurat, liger to be added, scalign needs to be deleted (?)