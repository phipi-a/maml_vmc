import jax.numpy as jnp

from maml_vmc.sampler.MoleculeDataSampler import (
    MoleculeFeatures,
)


class CuspCorrection:
    def __init__(
        self,
        r_cusp_el_el: float = 1.0,
    ):
        self.r_cusp_el_el = r_cusp_el_el

    def __call__(self, dist_el_el, mol: MoleculeFeatures) -> jnp.ndarray:
        # No factor 0.5 here, e.g. when comparing to NatChem 2020, [doi.org/10.1038/s41557-020-0544-y], because:
        # A) We double-count electron-pairs because we take the full distance matrix (and not only the upper triangle)
        # B) We model log(psi^2)=2*log(|psi|) vs log(|psi|) int NatChem 2020, i.e. the cusp correction needs a factor 2
        active_electrons = mol.active_electrons[:, None] * mol.active_electrons[None, :]
        same_spin = mol.electron_spins[:, None] == mol.electron_spins[None, :]
        factor = jnp.where(same_spin, 0.25, 0.5)
        factor = factor * active_electrons
        a = self.r_cusp_el_el**2
        b = self.r_cusp_el_el
        same_electron_mask = ~jnp.eye(dist_el_el.shape[0], dtype=bool)
        return -jnp.sum((factor * a / (dist_el_el + b) * same_electron_mask))
