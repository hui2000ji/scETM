from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import scipy


def entropy_batch_mixing(latent_space, batches, n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    # code adapted from scGAN
    def entropy(hist_data):
        counts = pd.Series(hist_data[:, 0]).value_counts()
        freqs = counts / counts.sum()
        return (-freqs * np.log(freqs)).sum()

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


print(entropy_batch_mixing(np.random.randn(30, 7),
                           np.random.randint(0, 2, size=(30, 1))))
