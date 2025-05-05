import jax
from maml_vmc import lijax
from maml_vmc.models.deep_erwin.utils import (
    get_distance_matrix_el_el,
    get_distance_matrix_el_ion,
)


from jax import numpy as jnp

from maml_vmc.sampler.MoleculeDataSampler import MoleculeFeatures


class FeatureExtraction(lijax.Module):
    def __init__(
        self,
        name: str = "FeatureExtraction",
        n_rbf_features: int = 32,
        distance_feature_powers: list = [-1],
    ):
        self.n_rbf_features = n_rbf_features
        self.distance_feature_powers = distance_feature_powers

        super().__init__(name=name)

    def init_modules(self):
        pass

    def __call__(self, x: MoleculeFeatures) -> jax.Array:
        # basic feature extraction
        el_pos = x.electron_positions.reshape(-1, 3)
        diff_el_el, dist_el_el = get_distance_matrix_el_el(el_pos, el_pos)
        diff_el_el = diff_el_el
        diff_el_ion, dist_el_ion = get_distance_matrix_el_ion(
            x.nuclear_positions,
            el_pos,
        )
        features_el_el = get_pairwise_features(
            dist_el_el, self.n_rbf_features, self.distance_feature_powers
        )
        features_el_ion = get_pairwise_features(
            dist_el_ion, self.n_rbf_features, self.distance_feature_powers
        )

        return (
            features_el_el,
            features_el_ion,
            diff_el_el,
            dist_el_el,
            diff_el_ion,
            dist_el_ion,
        )


# def get_distance_matrix(r_el):
#     """
#     Compute distance matrix omitting the main diagonal (i.e. distance to the particle itself)

#     Args:
#         r_el: [n_electrons x 3]

#     Returns:
#         tuple: differences [n_el x (n_el-1) x 3], distances [n_el x (n_el-1)]
#     """
#     n_el = r_el.shape[-2]
#     indices = jnp.array(
#         [[j for j in range(n_el) if j != i] for i in range(n_el)], dtype=int
#     )
#     diff = r_el[..., indices, :] - r_el[:, None, :]
#     dist = jnp.linalg.norm(diff, axis=-1)
#     return diff, dist


def get_rbf_features(dist, n_features):
    """
    Computes radial basis features based on Gaussians with different means from pairwise distances. This can be interpreted as a special type of "one-hot-encoding" for the distance between two particles.

    Args:
        dist (array): Pairwise particle distances
        n_features (int): Number of radial basis features
    Returns:
        array: Pairwise radial basis features

    """
    r_rbf_max = 5.0
    q = jnp.linspace(0, 1.0, n_features, dtype=dist.dtype)
    mu = q**2 * r_rbf_max
    sigma = (1 / 7) * (1 + r_rbf_max * q)
    dist = dist[..., jnp.newaxis]  # add dimension for features
    return dist**2 * jnp.exp(-dist - ((dist - mu) / sigma) ** 2)


def get_pairwise_features(dist, n_rbf_features, distance_feature_powers=[-1]):
    """
    Computes pairwise features based on particle distances.

    Args:
        dist (array): Pairwise particle distances
        dist_feat (bool): Flag that controls the usage of features of the form `dist`^n

    Returns:
        array: Pairwise distance features

    """
    features = get_rbf_features(dist, n_rbf_features)
    eps = 0.01
    if len(distance_feature_powers) > 0:
        f_r = jnp.stack(
            [
                dist**n if n > 0 else 1 / (dist ** (-n) + eps)
                for n in distance_feature_powers
            ],
            axis=-1,
        )
        features = jnp.concatenate([f_r, features], axis=-1)
    return features
