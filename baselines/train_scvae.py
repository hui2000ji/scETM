import argparse
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import scvae
from scvae import analyses
from scvae.analyses.prediction import (
    PredictionSpecifications, predict_labels
)
from scvae.data import DataSet
from scvae.data.utilities import (
    build_directory_path, indices_for_evaluation_subset
)
from scvae.defaults import defaults
from scvae.models import (
    VariationalAutoencoder,
    GaussianMixtureVariationalAutoencoder
)
from scvae.models.utilities import (
    better_model_exists, model_stopped_early,
    parse_model_versions
)
from scvae.utilities import (
    title, subtitle, heading,
    normalise_string, enumerate_strings,
    remove_empty_directories
)

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def train_and_eval(data_set,
          split_data_set=None, splitting_method=None, splitting_fraction=None,
          model_type=None, latent_size=None, hidden_sizes=None,
          number_of_importance_samples=None,
          number_of_monte_carlo_samples=None,
          inference_architecture=None, latent_distribution=None,
          number_of_classes=None, parameterise_latent_posterior=False,
          prior_probabilities_method=None,
          generative_architecture=None, reconstruction_distribution=None,
          number_of_reconstruction_classes=None, count_sum=None,
          proportion_of_free_nats_for_y_kl_divergence=None,
          minibatch_normalisation=None, batch_correction=None,
          dropout_keep_probabilities=None,
          number_of_warm_up_epochs=None, kl_weight=None,
          number_of_epochs=None, minibatch_size=None, learning_rate=None,
          run_id=None, new_run=False, reset_training=None,
          models_directory=None, caches_directory=None,
          analyses_directory=None, **keyword_arguments):
    """Train model on data set."""

    if models_directory is None:
        models_directory = defaults["models"]["directory"]

    print(title("Data"))

    binarise_values = False

    # data_set.load()
    splitting_method = None
    splitting_fraction = None
    training_set = data_set
    validation_set = None

    models_directory = build_directory_path(
        models_directory,
        data_set=data_set,
        splitting_method=splitting_method,
        splitting_fraction=splitting_fraction
    )

    if analyses_directory:
        analyses_directory = build_directory_path(
            analyses_directory,
            data_set=data_set,
            splitting_method=splitting_method,
            splitting_fraction=splitting_fraction
        )

    model_caches_directory = None
    if caches_directory:
        model_caches_directory = os.path.join(caches_directory, "log")
        model_caches_directory = build_directory_path(
            model_caches_directory,
            data_set=data_set,
            splitting_method=splitting_method,
            splitting_fraction=splitting_fraction
        )

    print(title("Model"))

    if number_of_classes is None:
        if training_set.has_labels:
            number_of_classes = (
                training_set.number_of_classes
                - training_set.number_of_excluded_classes)

    model = _setup_model(
        data_set=training_set,
        model_type=model_type,  # VAE/GMVAE
        latent_size=latent_size,  # 100
        hidden_sizes=hidden_sizes,  # [256, 128]
        number_of_importance_samples=number_of_importance_samples,  # 1
        number_of_monte_carlo_samples=number_of_monte_carlo_samples,  # 1
        inference_architecture=inference_architecture,  #  MLP
        latent_distribution=latent_distribution,  # "gaussian_mixture" or "full_covariance_gaussian_mixture"
        number_of_classes=number_of_classes,  #  1
        parameterise_latent_posterior=parameterise_latent_posterior,  #  False
        prior_probabilities_method=prior_probabilities_method,  # uniform, custom, or learn
        generative_architecture=generative_architecture,  # MLP
        reconstruction_distribution=reconstruction_distribution,  # poisson / negative binomial
        number_of_reconstruction_classes=number_of_reconstruction_classes,  # 0
        count_sum=count_sum,  # False
        proportion_of_free_nats_for_y_kl_divergence=(
            proportion_of_free_nats_for_y_kl_divergence),  # 0
        minibatch_normalisation=minibatch_normalisation,  # True
        batch_correction=batch_correction,  # True
        dropout_keep_probabilities=dropout_keep_probabilities,  # []
        number_of_warm_up_epochs=number_of_warm_up_epochs,  # 600
        kl_weight=kl_weight,  # 1.
        models_directory=models_directory  # model
    )

    print(model.description)
    print()

    print(model.parameters)
    print()

    print(subtitle("Training"))

    if analyses_directory:
        intermediate_analyser = analyses.analyse_intermediate_results
    else:
        intermediate_analyser = None

    model.train(
        training_set,
        validation_set,
        number_of_epochs=number_of_epochs,
        minibatch_size=minibatch_size,
        learning_rate=learning_rate,
        intermediate_analyser=intermediate_analyser,
        run_id=run_id,
        new_run=new_run,
        reset_training=reset_training,
        analyses_directory=analyses_directory,
        temporary_log_directory=model_caches_directory
    )

    # Remove temporary directories created and emptied during training
    if model_caches_directory and os.path.exists(caches_directory):
        remove_empty_directories(caches_directory)

    latent = model.evaluate(training_set, minibatch_size=minibatch_size, run_id=run_id, output_versions=['latent'])

    y, z = latent['y'].values, latent['z'].values

    label_pred = y.argmax(axis=1)

    return label_pred, z


def _setup_model(data_set, model_type=None,
                 latent_size=None, hidden_sizes=None,
                 number_of_importance_samples=None,
                 number_of_monte_carlo_samples=None,
                 inference_architecture=None, latent_distribution=None,
                 number_of_classes=None, parameterise_latent_posterior=False,
                 prior_probabilities_method=None,
                 generative_architecture=None,
                 reconstruction_distribution=None,
                 number_of_reconstruction_classes=None, count_sum=None,
                 proportion_of_free_nats_for_y_kl_divergence=None,
                 minibatch_normalisation=None, batch_correction=None,
                 dropout_keep_probabilities=None,
                 number_of_warm_up_epochs=None, kl_weight=None,
                 models_directory=None):

    if model_type is None:
        model_type = defaults["model"]["type"]
    if batch_correction is None:
        batch_correction = defaults["model"]["batch_correction"]

    feature_size = data_set.number_of_features
    number_of_batches = data_set.number_of_batches

    if not data_set.has_batches:
        batch_correction = False

    if normalise_string(model_type) == "vae":
        model = VariationalAutoencoder(
            feature_size=feature_size,
            latent_size=latent_size,
            hidden_sizes=hidden_sizes,
            number_of_monte_carlo_samples=number_of_monte_carlo_samples,
            number_of_importance_samples=number_of_importance_samples,
            inference_architecture=inference_architecture,
            latent_distribution=latent_distribution,
            number_of_latent_clusters=number_of_classes,
            parameterise_latent_posterior=parameterise_latent_posterior,
            generative_architecture=generative_architecture,
            reconstruction_distribution=reconstruction_distribution,
            number_of_reconstruction_classes=number_of_reconstruction_classes,
            minibatch_normalisation=minibatch_normalisation,
            batch_correction=batch_correction,
            number_of_batches=number_of_batches,
            dropout_keep_probabilities=dropout_keep_probabilities,
            count_sum=count_sum,
            number_of_warm_up_epochs=number_of_warm_up_epochs,
            kl_weight=kl_weight,
            log_directory=models_directory
        )

    elif normalise_string(model_type) == "gmvae":
        prior_probabilities_method_for_model = prior_probabilities_method
        if prior_probabilities_method == "uniform":
            prior_probabilities = None
        elif prior_probabilities_method == "infer":
            prior_probabilities_method_for_model = "custom"
            prior_probabilities = data_set.class_probabilities
        else:
            prior_probabilities = None

        model = GaussianMixtureVariationalAutoencoder(
            feature_size=feature_size,
            latent_size=latent_size,
            hidden_sizes=hidden_sizes,
            number_of_monte_carlo_samples=number_of_monte_carlo_samples,
            number_of_importance_samples=number_of_importance_samples,
            prior_probabilities_method=prior_probabilities_method_for_model,
            prior_probabilities=prior_probabilities,
            latent_distribution=latent_distribution,
            number_of_latent_clusters=number_of_classes,
            proportion_of_free_nats_for_y_kl_divergence=(
                proportion_of_free_nats_for_y_kl_divergence),
            reconstruction_distribution=reconstruction_distribution,
            number_of_reconstruction_classes=number_of_reconstruction_classes,
            minibatch_normalisation=minibatch_normalisation,
            batch_correction=batch_correction,
            number_of_batches=number_of_batches,
            dropout_keep_probabilities=dropout_keep_probabilities,
            count_sum=count_sum,
            number_of_warm_up_epochs=number_of_warm_up_epochs,
            kl_weight=kl_weight,
            log_directory=models_directory
        )

    else:
        raise ValueError("Model type not found: `{}`.".format(model_type))

    return model


def umap_and_leiden(adata, save_path=False, use_rep=None,
                    leiden_resolution=0.35, visualize_batch=True, show=False):
    print(f'\n========== Resolution {leiden_resolution} ==========')
    color=['batch_indices', 'leiden', 'cell_types'] if visualize_batch else ['leiden', 'cell_types']
    for item in ('condition', 'y'):
        if item in adata.obs:
            color.append(item)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=leiden_resolution)
    print(f'Resolution: {leiden_resolution}, # clusters: {adata.obs.leiden.nunique()}')
    print(f'ARI_type: {adjusted_rand_score(adata.obs.cell_types, adata.obs.leiden)}')
    print(f'NMI_type: {normalized_mutual_info_score(adata.obs.cell_types, adata.obs.leiden)}')
    if visualize_batch:
        print(f'ARI_batch: {adjusted_rand_score(adata.obs.batch_indices, adata.obs.leiden)}')
        print(f'NMI_batch: {normalized_mutual_info_score(adata.obs.batch_indices, adata.obs.leiden)}')
    sc.pl.umap(adata, color=color, use_raw=False, save=save_path, show=show)
    if show:
        plt.show()


from sklearn.neighbors import NearestNeighbors
import pandas as pd
import scipy
def entropy_batch_mixing(latent_space, batches, n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    # code adapted from scGAN
    print('Calculating batch mixing entropy...', end='\r', flush=True)
    def entropy(hist_data):
        counts = pd.Series(hist_data).value_counts()
        freqs = counts / counts.sum()
        return (-freqs * np.log(freqs + 1e-20)).sum()

    n_neighbors = min(n_neighbors, len(latent_space) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(latent_space)
    kmatrix = nne.kneighbors_graph(
        latent_space) - scipy.sparse.identity(latent_space.shape[0])

    score = 0.
    for t in range(n_pools):
        indices = np.random.choice(
            np.arange(latent_space.shape[0]), size=n_samples_per_pool)
        score += np.mean(
            [
                entropy(
                    batches[
                        kmatrix[indices[i]].nonzero()[1]
                    ]
                )
                for i in range(n_samples_per_pool)
            ]
        )
    return score / n_pools


if __name__ == '__main__':
    import anndata
    import argparse
    from pathlib import Path
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--h5ad-path', type=str, help="path to h5ad file storing the dataset")
    parser.add_argument('--model', type=str, choices=('VAE', 'GMVAE'), default='GMVAE', help="model to use")
    parser.add_argument('--latent-size', type=int, default=100, help="size of latent vector")
    parser.add_argument('--latent-distribution', type=str, choices=('full_covariance_gaussian_mixture', 'gaussian_mixture'), default='gaussian_mixture', help="latent distribution")
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=(256, 128), help="sizes of hidden layers as a list, whose length denote depth of the MLP")
    parser.add_argument('--n-labels', type=int, default=-1, help="number of classes in the dataset")
    parser.add_argument('--prior', type=str, choices=('uniform', 'custom', 'learn'), default='uniform', help="prior distribution")
    parser.add_argument('--reconstruction-distribution', type=str, choices=('poisson', 'zero-inflated poisson', 'negative binomial', 'zero-inflated negative binomial', 'lomax'), default='negative binomial', help="reconstruction distribution")
    parser.add_argument('--number-of-reconstruction-classes', type=int, default=0, help="the maximum count for which to use classification")
    parser.add_argument('--batch-removal', action='store_true', help="whether to add batch correction")
    parser.add_argument('--keep-probs', type=float, nargs='+', default=(1., 1., 1., 1.), help="dropout keep prob for h, x, z and y, respectively")
    parser.add_argument('--epochs', type=int, default=500, help="number of epochs to train")
    parser.add_argument('--warm-up-epochs', type=int, default=200, help="number of warm-up epochs")
    parser.add_argument('--kl-weight', type=float, default=1., help="weight of KL in VAE loss")
    parser.add_argument('--ckpt-dir', type=str, default='../results', help="directory to store checkpoints")
    parser.add_argument('--batch-size', type=int, default=250, help="batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate of the model")
    parser.add_argument('--no-restore', action='store_true', help="do not restore trained model even if possible")
    parser.add_argument('--resolutions', type=float, nargs='+', default=(0.02, 0.04, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7), help="leiden resolutions")
    parser.add_argument('--run-id', type=str, default='', help="a string to distinguish different runs")
    args = parser.parse_args()

    adata = anndata.read_h5ad(args.h5ad_path)
    dataset_str = Path(args.h5ad_path).stem
    if args.batch_removal:
        dataset_str = dataset_str + '_batch'

    col = list(map(lambda s: s.lower(), list(adata.obs.columns)))
    adata.obs.columns = col
    if 'cell_type' in col:
        col[col.index('cell_type')] = 'cell_types'
    convert_batch_to_int = False
    if 'batch_id' in col:
        batches = list(adata.obs.batch_id.unique())
        batches.sort()
        if not isinstance(batches[-1], str) and batches[-1] + 1 == len(batches):
            col[col.index('batch_id')] = 'batch_indices'
        else:
            convert_batch_to_int = True
    adata.obs.columns = col
    if convert_batch_to_int:
        adata.obs['batch_indices'] = adata.obs.batch_id.apply(lambda x: batches.index(x))

    if args.n_labels == -1:
        args.n_labels = adata.obs.cell_types.nunique()

    from scipy.sparse import csr_matrix
    data_set = DataSet(dataset_str,
        title=dataset_str,
        specifications=dict(),
        values=csr_matrix(adata.X),
        labels=np.asarray_chkfinite(adata.obs.cell_types),
        batch_indces=np.asarray_chkfinite(adata.obs.batch_indices),
        feature_names=np.asarray_chkfinite(adata.var_names)
    )

    labels, latent = train_and_eval(data_set,
        model_type=args.model,
        latent_size=args.latent_size,
        latent_distribution=args.latent_distribution,
        hidden_sizes=args.hidden_sizes,
        number_of_classes=args.n_labels,
        number_of_epochs=args.epochs,
        number_of_warm_up_epochs=args.warm_up_epochs,
        prior_probabilities_method=args.prior,
        reconstruction_distribution=args.reconstruction_distribution,
        number_of_reconstruction_classes=args.number_of_reconstruction_classes,
        batch_correction=args.batch_removal,
        dropout_keep_probabilities=args.keep_probs,
        kl_weight=args.kl_weight,
        models_directory=args.ckpt_dir,
        reset_training=args.no_restore,
        minibatch_size=args.batch_size,
        learning_rate=args.lr
    )

    adata.obs['y'] = labels
    print(f'ARI_type: {adjusted_rand_score(adata.obs.cell_types, adata.obs.y)}')
    print(f'NMI_type: {normalized_mutual_info_score(adata.obs.cell_types, adata.obs.y)}')
    if adata.obs.batch_indices.nunique() > 1:
        print(f'ARI_batch: {adjusted_rand_score(adata.obs.batch_indices, adata.obs.y)}')
        print(f'NMI_batch: {normalized_mutual_info_score(adata.obs.batch_indices, adata.obs.y)}')
    
    adata.obsm['latent'] = latent
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="latent")
    for resolution in args.resolutions:
        umap_and_leiden(adata, save_path=f'_{dataset_str}_scVAE_{resolution}.pdf', use_rep='latent', leiden_resolution=resolution)
        if adata.obs.leiden.nunique() > adata.obs.cell_types.nunique():
            break
    
    print(f'BE: {entropy_batch_mixing(latent, adata.obs.batch_indices)}')
