import argparse
import anndata
from pathlib import Path

from scETM import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--h5ad-path')
args = parser.parse_args()

adata = anndata.read_h5ad(args.h5ad_path)
evaluate(adata, embedding_key='X', resolutions=[0.1, 0.15, 0.22, 0.34, 0.51, 0.76], plot_dir=str(Path(args.h5ad_path).parent))
