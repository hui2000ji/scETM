import argparse
import os

parser = argparse.ArgumentParser()

# Program behavior
parser.add_argument('--eval', action='store_true', help='eval mode')

# Model parameters
parser.add_argument('--model', type=str, default='scETM', help="model name")
parser.add_argument('--device', type=str, default='cuda', help="device to hold the model")
parser.add_argument('--hidden-sizes', type=int, nargs='+', default=(128,), help='Hidden sizes of the encoder')
parser.add_argument('--dropout-prob', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--trainable-gene-emb-dim', type=int, default=400, help='gene embedding dimensions')
parser.add_argument('--no-bn', action='store_true', help='Disable batch normalization')
parser.add_argument('--n-topics', type=int, default=50, help='number of topics in model')
parser.add_argument('--no-batch-bias', action='store_false', dest='batch_bias', help='enable batch-specific bias')
parser.add_argument('--global-bias', action='store_true', help='enable global gene bias')
parser.add_argument('--adv-loss', choices=('reverse', 'confuse'), default='reverse')

# Loss parameters
parser.add_argument('--g-steps', type=int, default=1)
parser.add_argument('--d-steps', type=int, default=8)
parser.add_argument('--max-kl-weight', type=float, default=1e-7, help='max weight for kl divergence')
parser.add_argument('--max-clf-weight', type=float, default=0.08, help='max weight for model loss')
parser.add_argument('--max-mmd-weight', type=float, default=1, help='max weight for mmd regumarization')
parser.add_argument('--min-kl-weight', type=float, default=0, help='min weight for kl divergence')
parser.add_argument('--min-clf-weight', type=float, default=0, help='min weight for model loss')
parser.add_argument('--min-mmd-weight', type=float, default=0, help='min weight for mmd regumarization')
parser.add_argument('--kl-warmup-ratio', type=float, default=1/3, help='gradually increase weight of the kl divergence loss during the first args.kl_warmup_ratio training epochs')
parser.add_argument('--clf-warmup-ratio', type=float, default=1/3, help='gradually increase weight of the clf loss during the first args.clf_warmup_ratio training epochs')
parser.add_argument('--clf-cutoff-ratio', type=float, default=1/6, help='disable clf loss during the first args.clf_cutoff_ratio training epochs')
parser.add_argument('--mmd-warmup-ratio', type=float, default=1/6, help='gradually increase weight of the mmd regularization during the first args.mmd_warmup_ratio training epochs')
parser.add_argument('--no-normed-loss', action='store_false', dest='normed_loss', help='whether to normalize gene expression when calculating loss')
parser.add_argument('--no-norm-cells', action='store_false', dest='norm_cells', help='whether to normalize gene expression at input')

# Training parameters
parser.add_argument('--seed', type=int, default=-1, help='Random seed')
parser.add_argument('--data-split-seed', type=int, default=1, help='data split random seed')
parser.add_argument('--n-epochs', type=int, default=2400, help='Number of epochs to train')
parser.add_argument('--eval-every', type=int, default=400, help='Number of epochs between loggings')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate')
parser.add_argument('--batch-clf-lr', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--lr-decay', type=float, default=5e-5, help='Negative log of the learning rate decay rate')
parser.add_argument('--batch-clf-lr-decay', type=float, default=5e-5, help='Negative log of the learning rate decay rate')
parser.add_argument('--test-ratio', type=float, default=0., help='Amount of held out data used for evaluation')
parser.add_argument('--batch-size', type=int, default=2000, help='Batch size for training')
parser.add_argument('--input-batch-id', action='store_true', help='concatenate batch indices to the input to the model')
parser.add_argument('--n-samplers', type=int, default=4, help='number of sampler thread')
parser.add_argument('--be', action='store_false', dest='no_be', help='calculate batch mixing entropy, which is very time consuming')
parser.add_argument('--no-eval', action='store_true', help='only do training, do not evaluate')

# Model save/restore parameters
parser.add_argument('--restore-epoch', type=int, default=0, help='epoch number of the checkpoint you wish to restore')
parser.add_argument('--ckpt-dir', type=str, default=os.path.join('..', 'results'), help='directory of checkpoints')
parser.add_argument('--log-str', type=str, default='', help='additional string on ckpt dir name')
parser.add_argument('--result-tsv', type=str, default='', help='path to the csv file logging the results')
parser.add_argument('--no-model-ckpt', action='store_true', help='do not checkpoint the model or the optimizer')

# Dataset location parameters
parser.add_argument('--dataset-str', type=str, default='cortex', help='dataset name. Must be the key of available_datasets')
parser.add_argument('--target-dataset-str', type=str, default='cortex', help='target dataset name. Must be the key of available_datasets')
parser.add_argument('--h5ad-path', type=str, default='', help='path to the h5ad file representing an Anndata object')
parser.add_argument('--target-h5ad-path', type=str, default='', help='path to the h5ad file representing the target Anndata object')
parser.add_argument('--pathway-csv-path', type=str, default='', help='path to the csv file containing the gene x pathway matrix')

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
    parser.add_argument('--color-by', nargs='*', default=['batch_indices', 'cell_types'],
                        help='columns of adata.obs that will be visualized at each evaluation')
    parser.add_argument('--n-neighbors', type=int, default=15, help='number of neighbors to compute UMAP')
    parser.add_argument('--min_dist', type=float, default=0.3, help='minimum distance b/t UMAP embedded points')
    parser.add_argument('--spread', type=float, default=1., help='scale of the embedded points')
    parser.add_argument('--point-size', type=float, default=0., help='size of the scatterpoints, if 0., defaults to 120k / n_cells')
    # clustering
    parser.add_argument('--clustering-input', type=str, default='delta', choices=('theta', 'delta'), help="input of batch classifier")
    parser.add_argument('--clustering-method', type=str, choices=('louvain', 'leiden'), default='leiden', help='clustering algorithm')
    parser.add_argument('--resolutions', type=float, nargs='+', default=(0.02, 0.04, 0.06, 0.09, 0.15, 0.23, 0.32, 0.45), help='resolution of leiden/louvain clustering')
    parser.add_argument('--fix-resolutions', action='store_true', help='do not automatically tune the resolutions')
    # sc.settings.set_figure_params
    parser.add_argument('--figsize', type=int, nargs=2, default=(10, 10), help='size of the plotted figure')
    parser.add_argument('--fontsize', type=int, default=10, help='font size in plotted figure')
    parser.add_argument('--dpi-show', type=int, default=120, help='Resolution of shown figures')
    parser.add_argument('--dpi-save', type=int, default=300, help='Resolution of saved figures')
add_plotting_arguments(parser)