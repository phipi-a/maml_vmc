UP = 1
DOWN = -1

from typing import NamedTuple
import jax.numpy as jnp


def remove_diagonal(a):
    # remove diagonal shape: (n,n,...) -> (n,n-1,...)
    a = a[~jnp.eye(a.shape[0], dtype=bool)].reshape(
        a.shape[0], a.shape[1] - 1, *a.shape[2:]
    )
    return a


def get_distance_matrix_el_el(r1, r2):
    """
    Args:
        r_el: shape [ n_el x 3]
        R_ion: shape [N_ion x 3]
    Returns:
        diff: shape [n_el x N_ion x 3]
        dist: shape [n_el x N_ion]
    """
    diff = r2 - r1[:, None, :]
    diff_el_el_add = (
        diff + jnp.eye(r1.shape[0], dtype=r1.dtype)[:, :, None]
    )  # add identity matrix
    dist = jnp.linalg.norm(diff_el_el_add, axis=-1)
    return diff, dist


def get_distance_matrix_el_ion(r1, r2):
    """
    Args:
        r_el: shape [ n_el x 3]
        R_ion: shape [N_ion x 3]
    Returns:
        diff: shape [n_el x N_ion x 3]
        dist: shape [n_el x N_ion]
    """
    diff = r2[:, None, :] - r1
    dist = jnp.linalg.norm(diff, axis=-1)
    return diff, dist


class MMSMoleculeFeaturesBaseline(NamedTuple):
    """Features of the system"""

    electron_positions: jnp.ndarray
    electron_spins: jnp.ndarray
    nuclear_positions: jnp.ndarray
    nuclear_charges: jnp.ndarray
    num_electrons: jnp.ndarray
    random_key: jnp.ndarray
    baseline_params: tuple
