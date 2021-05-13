import argparse
import anndata
from pathlib import Path

from scETM import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--h5ad-path')
parser.add_argument('--resolutions', type=float, nargs='+', default=[0.75, 1, 1.3, 1.6, 2])
args = parser.parse_args()

adata = anndata.read_h5ad(args.h5ad_path)
evaluate(adata, embedding_key='X', resolutions=args.resolutions, plot_dir=str(Path(args.h5ad_path).parent))
