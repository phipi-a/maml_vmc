import jax

from maml_vmc import lijax
from maml_vmc.models.deep_erwin.orbitals import evaluate_molecular_orbitals
import jax.numpy as jnp

from maml_vmc.sampler.MoleculeDataSampler import (
    MoleculeFeatures,
)


class CASSCFBaseline(lijax.Module):
    # def __init__(
    #     self,
    # ):
    #     pass

    def __call__(self, diff_el_ion, dist_el_ion, mol: MoleculeFeatures) -> jax.Array:
        mo_matrix_up = evaluate_molecular_orbitals(
            diff_el_ion,
            dist_el_ion,
            mol.baseline_params.atomic_orbitals,
            mol.baseline_params.mo_coeff_up,
            mol.baseline_params.cusp_params_up,
            mol.active_electrons,
        )

        mo_matrix_dn = evaluate_molecular_orbitals(
            diff_el_ion,
            dist_el_ion,
            mol.baseline_params.atomic_orbitals,
            mol.baseline_params.mo_coeff_dn,
            mol.baseline_params.cusp_params_dn,
            mol.active_electrons,
        )

        # select used orbitals for each electron (number of selected orbitals per electron == number of electrons)
        mo_matrix_dn1 = jnp.roll(mo_matrix_dn, shift=-mol.n_up, axis=0)
        mo_matrix_up2 = mo_matrix_up[..., mol.baseline_params.ind_orb_up]
        mo_matrix_dn2 = mo_matrix_dn1[..., mol.baseline_params.ind_orb_dn]
        # electron x ndeterminants x norbs -> ndeterminants x electron x norbs
        mo_matrix_up3 = jnp.transpose(mo_matrix_up2, [1, 0, 2])
        mo_matrix_dn3 = jnp.transpose(mo_matrix_dn2, [1, 0, 2])
        mo_matrix_up3 = mo_matrix_up3 * mol.active_electrons[None, None, :]
        mo_matrix_dn3 = mo_matrix_dn3 * mol.active_electrons[None, None, :]
        # n_det x n_dn + n_up x n_dn+n_up
        return mo_matrix_up3, mo_matrix_dn3, mol.baseline_params.ci_weights
