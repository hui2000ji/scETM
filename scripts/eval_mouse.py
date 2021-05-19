try:
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except:
    pass

import argparse
import anndata
import matplotlib
import scanpy as sc
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from scETM import evaluate

matplotlib.use('Agg')
sc.set_figure_params(figsize=(10, 10), fontsize=10, dpi=120, dpi_save=250)

parser = argparse.ArgumentParser()
parser.add_argument('--h5ad-path')
parser.add_argument('--resolutions', type=float, nargs='+', default=[0.75, 1])
args = parser.parse_args()

d = Path(args.h5ad_path).parent
adata = anndata.read_h5ad(args.h5ad_path)
writer = SummaryWriter(str(d / 'tensorboard'))
evaluate(adata, embedding_key='X', resolutions=args.resolutions, plot_dir=str(d), writer=writer)
