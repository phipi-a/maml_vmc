import jax

import jax.numpy as jnp
from matplotlib.sankey import UP

from maml_vmc.models.deep_erwin.utils import DOWN
from maml_vmc.sampler.MoleculeDataSampler import (
    MoleculeFeatures,
)
import haiku as hk


class IsoEnv:
    def __init__(
        self,
        max_ions: int,
        max_electrons: int,
        n_dets: int = 32,
    ):
        self.max_ions = max_ions
        self.max_electrons = max_electrons
        self.n_dets = n_dets

    def __call__(self, diff_el_ion, dist_el_ion, mol: MoleculeFeatures) -> jax.Array:
        # el_ion_dist: electron x ions
        alpha_up = hk.get_parameter(
            "alpha_up",
            shape=[self.max_ions, self.n_dets, self.max_electrons],
            init=jnp.ones,
        )
        alpha_dn = hk.get_parameter(
            "alpha_down",
            shape=[self.max_ions, self.n_dets, self.max_electrons],
            init=jnp.ones,
        )
        weights_up = hk.get_parameter(
            "weights_up",
            shape=[self.max_ions, self.n_dets, self.max_electrons],
            init=jnp.ones,
        )
        weights_dn = hk.get_parameter(
            "weights_down",
            shape=[self.max_ions, self.n_dets, self.max_electrons],
            init=jnp.ones,
        )
        ci_weights = jnp.ones(self.n_dets)
        # alpha_up: ions x ndeterminants x electrons
        # 1x ions x dets x el * el x ions x 1 x 1
        exp_up = (
            jax.nn.softplus(alpha_up[None, :, :, :]) * dist_el_ion[:, :, None, None]
        )
        exp_dn = (
            jax.nn.softplus(alpha_dn[None, :, :, :]) * dist_el_ion[:, :, None, None]
        )
        # exp_up: el x ions x dets x el
        ion_mask = mol.active_nuclears[None, :, None, None]
        orb_up = jnp.sum(
            jnp.exp(-exp_up) * weights_up[None, :, :, :] * ion_mask, axis=-3
        )
        orb_dn = jnp.sum(
            jnp.exp(-exp_dn) * weights_dn[None, :, :, :] * ion_mask, axis=-3
        )
        # orb_up: el x dets x el
        orb_up = (mol.electron_spins == UP)[:, None, None] * orb_up
        orb_dn = (mol.electron_spins == DOWN)[:, None, None] * orb_dn

        orb_dn = jnp.roll(orb_dn, shift=-mol.n_up, axis=0)

        # select used orbitals for each electron (number of selected orbitals per electron == number of electrons)
        # mo_matrix_dn1 = jnp.roll(orb_up, shift=-mol.n_up, axis=0)
        # electron x ndeterminants x norbs -> ndeterminants x electron x norbs
        mo_matrix_up3 = jnp.transpose(orb_up, [1, 0, 2])
        mo_matrix_dn3 = jnp.transpose(orb_dn, [1, 0, 2])
        mo_matrix_up3 = mo_matrix_up3 * mol.active_electrons[None, None, :]
        mo_matrix_dn3 = mo_matrix_dn3 * mol.active_electrons[None, None, :]

        return mo_matrix_up3, mo_matrix_dn3, ci_weights
