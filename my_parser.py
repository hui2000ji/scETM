import argparse
import os

parser = argparse.ArgumentParser()

# Program behavior
parser.add_argument('--eval', action='store_true', help='eval mode')

# Model parameters
parser.add_argument('--model', type=str, default='scETM', help="models used")
parser.add_argument('--hidden-sizes', type=int, nargs='+', default=(256, 128), help='Hidden sizes of theencoder')
parser.add_argument('--dropout-prob', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--gene-emb-dim', type=int, default=300, help='gene embedding dimensions')
parser.add_argument('--no-bn', action='store_true', help='Disable batch normalization')
parser.add_argument('--n-topics', type=int, default=100, help='number of topics in model')
parser.add_argument('--norm-cells', action='store_true', help='normalize cell samples')
parser.add_argument('--mask-ratio', type=float, default=0.2, help='random masking ratio of gene expression')
parser.add_argument('--batch-scaling', action='store_true', help='enable cell- and batch-specific scaling')

# Loss parameters
parser.add_argument('--max-supervised-weight', type=float, default=0, help='weight of supervsied loss, 0 to turn off supervised components')
parser.add_argument('--max-kl-weight', type=float, default=1., help='max weight for kl divergence')
parser.add_argument('--min-kl-weight', type=float, default=0., help='min weight for kl divergence')
parser.add_argument('--kl-weight-anneal', type=int, default=300, help='linear annealing of kl divergence loss')
parser.add_argument('--normed-loss', action='store_true', help='whether to normalize gene expression when calculating loss')

# Training parameters
parser.add_argument('--seed', type=int, default=-1, help='Random seed')
parser.add_argument('--n-epochs', type=int, default=800, help='Number of epochs to train')
parser.add_argument('--updates', type=int, help="Number of updates to train (depracated)")
parser.add_argument('--log-every', type=int, default=100, help='Number of epochs between loggings')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate')
parser.add_argument('--lr-decay', type=float, default=6e-5, help='Negative log of the learning rate decay rate')
parser.add_argument('--batch-size', type=int, default=2000, help='Batch size for training')
parser.add_argument('--input-batch-id', action='store_true', help='concatenate batch indices to the input to the model')
parser.add_argument('--n-samplers', type=int, default=4, help='number of sampler thread')
parser.add_argument('--no-be', action='store_true', help='do not calculate batch mixing entropy, which is very time consuming')
parser.add_argument('--no-eval', action='store_true', help='only do training, do not evaluate')

# Model save/restore parameters
parser.add_argument('--restore-step', type=int, help='step of the checkpoint you wish to restore (depracated)')
parser.add_argument('--restore-epoch', type=int, default=0, help='epoch number of the checkpoint you wish to restore')
parser.add_argument('--ckpt-dir', type=str, default=os.path.join('..', 'results'), help='directory of checkpoints')
parser.add_argument('--log-str', type=str, default='', help='additional string on ckpt dir name')
parser.add_argument('--tracked-metric', type=str, default='l_nmi', help='metric to track for auto ckpt deletion')

# Dataset location parameters
parser.add_argument('--dataset-str', type=str, default='cortex', help='dataset name')
parser.add_argument('--dataframe-path', type=str, default='', help='path to pickled pandas dataframe')
parser.add_argument('--anndata-path', type=str, default='', help='path to pickled Anndata object')
parser.add_argument('--h5ad-path', type=str, default='', help='path to the h5ad file representing an Anndata object')

# Dataset preprocessing parameters
parser.add_argument('--clip', type=int, default=0, help='enable dataset clipping, 0 for not clipping')
parser.add_argument('--quantile-norm', action='store_true', help='enable quantile normalization for cell-gene matrix')
parser.add_argument('--log1p', action='store_true', help='log1p transform the dataset')
parser.add_argument('--norm-cell-read-counts', action='store_true', help='whether to normalize cell read counts')

# Embedding plotting parameters
parser.add_argument('--always-draw', nargs='*', default=['cell_types', 'leiden', 'batch_indices'],
                    help='embeddings that will be drawn after each evaluation (gt, p, q, k, batch)')
parser.add_argument('--n-neighbors', type=int, default=15, help='number of neighbors to compute UMAP')
parser.add_argument('--min_dist', type=float, default=0.3, help='minimum distance b/t UMAP embedded points')
parser.add_argument('--spread', type=float, default=1., help='scale of the embedded points')
parser.add_argument('--louvain-resolutions', type=float, nargs='*', default=(0.05, 0.1, 0.15, 0.2), help='resolution of louvain clustering')
parser.add_argument('--leiden-resolutions', type=float, nargs='*', default=(0.05, 0.1, 0.15, 0.2), help='resolution of leiden clustering')
parser.add_argument('--group-by', type=str, default='condition', help='dataframe (adata.obs) column to group the cells by, used in scETMMultiDecoder')
args = parser.parse_args()