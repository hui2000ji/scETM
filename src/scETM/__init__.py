"""Single-cell Embedded topic model.

An accurate, transferrable and interpretable model with highly scalable and
easy-to-use APIs for single-cell RNA sequencing data analysis. Fully integrated
with scanpy and anndata.

From paper "Learning interpretable cellular and gene signature embeddings from
single-cell transcriptomic data".
Link: https://www.biorxiv.org/content/10.1101/2021.01.13.426593v1.full

If you have any problems using this package, please refer to our project site
https://www.github.com/hui2000ji/scETM.
"""

from scETM.logging_utils import initialize_logger
from scETM.models import scETM, scVI
from scETM.trainers import UnsupervisedTrainer, MMDTrainer, BatchAdversarialTrainer, prepare_for_transfer, train_test_split, set_seed
from scETM.eval_utils import evaluate, calculate_entropy_batch_mixing, calculate_kbet, clustering, draw_embeddings, set_figure_params

initialize_logger()