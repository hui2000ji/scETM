import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vGraph', help="models used")
# parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--updates', type=int, default=9600, help='Number of updates to train.')
parser.add_argument('--eval-batches', type=int, default=0, help='Number of batches in eval')
parser.add_argument('--log-every', type=int, default=1200, help='Display and write logs every such steps.')
parser.add_argument('--emb-dim', type=int, default=128, help='')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
parser.add_argument('--lr-decay', type=float, default=1., help='Learning rate decay rate')
# parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cortex', help='dataset name')
parser.add_argument('--batch-size', type=int, default=80000, help='Batch size for training')
parser.add_argument('--ckpt-dir', type=str, default='../results', help='directory of checkpoints')
parser.add_argument('--restore-step', type=int, default=0, help='k-step of the checkpoint you wish to restore')
parser.add_argument('--emb-combine', type=str, default='prod',
                    help='the way to combine cell and gene embeddings into cell-type embeddings: prod, sum, cat')
parser.add_argument('--clip', type=int, default=0, help='enable dataset clipping, 0 for not clipping')
parser.add_argument('--scale', type=float, default=0., help='enable dataset scaling, 0 for not scaling')
parser.add_argument('--subsample-genes', type=int, default=-1,
                    help='no. of genes with highest variance to keep, 0 indicates all')
parser.add_argument('--quantile-norm', action='store_true', help='enable quantile normalization for cell-gene matrix')
parser.add_argument('--log1p', action='store_true', help='log1p transform the dataset')
parser.add_argument('--max-beta', type=float, default=1., help='max weight for kl divergence')
parser.add_argument('--min-beta', type=float, default=0., help='min weight for kl divergence')
parser.add_argument('--max-gamma', type=float, default=0., help='max weight for LINE gg reconstruction')
parser.add_argument('--max-delta', type=float, default=0., help='max weight for LINE cc reconstruction')
parser.add_argument('--max-epsilon', type=float, default=0., help='max weight for LINE cg/gc reconstruction')
parser.add_argument('--max-zeta', type=float, default=0., help='max weight for vGraph word node reconstruction')
parser.add_argument('--max-eta', type=float, default=0., help='max weight for cell type center distancing')
parser.add_argument('--max-lambda', type=float, default=0., help='max weight for mutual information of batch and inferred cell type')
parser.add_argument('--cyclic-anneal', type=int, default=0, help='cyclic annealing of beta term')
parser.add_argument('--linear-anneal', type=int, default=1200, help='linear annealing of beta term')
parser.add_argument('--linear-anneal-eta', type=int, default=0, help='linear annealing of cell type center distancing term')
parser.add_argument('--linear-anneal-epsilon', type=int, default=0, help='linear annealing of LINE loss')
parser.add_argument('--bind-emb', action='store_true', help='bind input and output embeddings')
parser.add_argument('--bind-wc', action='store_true', help='bind word and context embeddings')
parser.add_argument('--gumbel', action='store_true', help='whether to use gumbel softmax')
parser.add_argument('--decouple-pq', action='store_true', help='decouple cell type embeddings in pq calculation')
parser.add_argument('--cell-type-dec', action='store_true', help='decouple cell type encoder and decoder')
parser.add_argument('--norm-cell-read-counts', action='store_true', help='whether to normalize cell read counts')
parser.add_argument('--always-draw', nargs='*', default=['cell_types', 'q', 'batch_indices'],
                    help='embeddings that will be drawn after each evaluation (gt, p, q, k, batch)')
parser.add_argument('--n-labels', type=int, default=0, help='number of labels in model')
parser.add_argument('--log-str', type=str, default='', help='additional string on ckpt dir name')
parser.add_argument('--tracked-metric', type=str, default='q_nmi', help='metric to track for auto ckpt deletion')
parser.add_argument('--g2c-factor', type=float, default=1., help='ratio between g2c and c2g losses. '
                                                                 'Only valid for vGraph2 / TwoWayCellGeneModel')
parser.add_argument('--neg-power', type=float, default=0.75, help='Power parameter in negative sampling')
parser.add_argument('--neg-samples', type=int, default=5, help='Number of negative samplers per training sample')
parser.add_argument('--neg-weight', type=float, default=1, help="weight of negative sample weights")
parser.add_argument('--gumbel-max', type=float, default=1., help='Initial (Maximum) gumbel tau')
parser.add_argument('--gumbel-min', type=float, default=0.3, help='Minimum gumbel tau')
parser.add_argument('--gumbel-anneal', type=float, default=0.00005, help='Negative log of multiplicative coefficient of gumbel tau')
parser.add_argument('--n-neighbors', type=int, default=15, help='number of neighbors to compute UMAP')
parser.add_argument('--figsize', nargs=2, default=(10, 10), help='figure size')
parser.add_argument('--min_dist', type=float, default=0.3, help='minimum distance b/t UMAP embedded points')
parser.add_argument('--spread', type=float, default=1., help='scale of the embedded points')
parser.add_argument('--alias-sampling', action='store_false', help='enable Vose Alias Sampling')
parser.add_argument('--encoder-depth', type=int, default=1, help='depth of the encoder')
parser.add_argument('--decoder-depth', type=int, default=1, help='depth of the decoder')