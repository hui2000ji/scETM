import argparse
import os

parser = argparse.ArgumentParser()

# Program behavior
parser.add_argument('--eval', action='store_true', help='eval mode')

# Model parameters
parser.add_argument('--model', type=str, default='scETM', help="model name")
parser.add_argument('--hidden-sizes', type=int, nargs='+', default=(256, 128), help='Hidden sizes of the encoder')
parser.add_argument('--dropout-prob', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--gene-emb-dim', type=int, default=300, help='gene embedding dimensions')
parser.add_argument('--no-bn', action='store_true', help='Disable batch normalization')
parser.add_argument('--n-topics', type=int, default=100, help='number of topics in model')
parser.add_argument('--norm-cells', action='store_true', help='normalize cell samples')
parser.add_argument('--mask-ratio', type=float, default=0.2, help='random masking ratio of gene expression')
parser.add_argument('--batch-scaling', action='store_true', help='enable batch-specific scaling')
parser.add_argument('--global-bias', action='store_true', help='enable global gene bias')

# Loss parameters
parser.add_argument('--max-supervised-weight', type=float, default=0, help='weight of supervsied loss, 0 to turn off supervised components')
parser.add_argument('--max-kl-weight', type=float, default=1., help='max weight for kl divergence')
parser.add_argument('--min-kl-weight', type=float, default=0., help='min weight for kl divergence')
parser.add_argument('--n-warmup-epochs', type=int, default=300, help='linear annealing of kl divergence loss')
parser.add_argument('--normed-loss', action='store_true', help='whether to normalize gene expression when calculating loss')

# Training parameters
parser.add_argument('--seed', type=int, default=-1, help='Random seed')
parser.add_argument('--n-epochs', type=int, default=800, help='Number of epochs to train')
parser.add_argument('--log-every', type=int, default=400, help='Number of epochs between loggings')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate')
parser.add_argument('--lr-decay', type=float, default=6e-5, help='Negative log of the learning rate decay rate')
parser.add_argument('--batch-size', type=int, default=2000, help='Batch size for training')
parser.add_argument('--input-batch-id', action='store_true', help='concatenate batch indices to the input to the model')
parser.add_argument('--n-samplers', type=int, default=4, help='number of sampler thread')
parser.add_argument('--no-be', action='store_true', help='do not calculate batch mixing entropy, which is very time consuming')
parser.add_argument('--no-eval', action='store_true', help='only do training, do not evaluate')

# Model save/restore parameters
parser.add_argument('--restore-epoch', type=int, default=0, help='epoch number of the checkpoint you wish to restore')
parser.add_argument('--ckpt-dir', type=str, default=os.path.join('..', 'results'), help='directory of checkpoints')
parser.add_argument('--log-str', type=str, default='', help='additional string on ckpt dir name')
parser.add_argument('--tracked-metric', type=str, default='leiden_ari', help='metric to track for auto ckpt deletion')
parser.add_argument('--save-embeddings', action='store_true', help='store cell, gene, topic embeddings after evaluation')

# Dataset location parameters
parser.add_argument('--dataset-str', type=str, default='cortex', help='dataset name. Must be the key of available_datasets')
parser.add_argument('--anndata-path', type=str, default='', help='path to pickled Anndata object')
parser.add_argument('--h5ad-path', type=str, default='', help='path to the h5ad file representing an Anndata object')
parser.add_argument('--pathway-csv-path', type=str, default='', help='path to the csv file containing the gene x pathway matrix')

# Dataset preprocessing parameters
def add_preprocessing_arguments(parser):
    """
    Add parameters '--clip', '--quantile-norm', '--log1p', '--norm-cell-read-counts'.
    """
    parser.add_argument('--clip', type=int, default=0, help='enable dataset clipping, 0 for not clipping')
    parser.add_argument('--quantile-norm', action='store_true', help='enable quantile normalization for cell-gene matrix')
    parser.add_argument('--log1p', action='store_true', help='log1p transform the dataset')
    parser.add_argument('--norm-cell-read-counts', action='store_true', help='whether to normalize cell read counts')
add_preprocessing_arguments(parser)

# Embedding plotting parameters
def add_plotting_arguments(parser):
    """
    Add parameters '--no_draw'; '--color_by', '--n_neighbors', '--min_dist', '--spread';
    '--clustering-method', '--resolutions', '--fix-resolutions'; '--figsize', '--fontsize',
    '--dpi-show', '--dpi-save'.
    """
    # plot or not
    parser.add_argument('--no-draw', action='store_true', help='do not draw')
    # draw_embeddings
    parser.add_argument('--color-by', nargs='*', default=['cell_types', 'batch_indices'],
                        help='columns of adata.obs that will be visualized at each evaluation')
    parser.add_argument('--n-neighbors', type=int, default=15, help='number of neighbors to compute UMAP')
    parser.add_argument('--min_dist', type=float, default=0.3, help='minimum distance b/t UMAP embedded points')
    parser.add_argument('--spread', type=float, default=1., help='scale of the embedded points')
    # clustering
    parser.add_argument('--clustering-method', type=str, choices=('louvain', 'leiden'), default='leiden', help='clustering algorithm')
    parser.add_argument('--resolutions', type=float, nargs='+', default=(0.05, 0.1, 0.15, 0.2), help='resolution of leiden/louvain clustering')
    parser.add_argument('--fix-resolutions', action='store_true', help='do not automatically tune the resolutions')
    # sc.settings.set_figure_params
    parser.add_argument('--figsize', type=int, nargs=2, default=(10, 10), help='size of the plotted figure')
    parser.add_argument('--fontsize', type=int, default=10, help='font size in plotted figure')
    parser.add_argument('--dpi-show', type=int, default=120, help='Resolution of shown figures')
    parser.add_argument('--dpi-save', type=int, default=250, help='Resolution of saved figures')
add_plotting_arguments(parser)