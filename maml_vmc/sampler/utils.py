from typing import NamedTuple
import jax
import jax.numpy as jnp


class CuspParams(NamedTuple):
    r_c: jnp.ndarray
    offset: jnp.ndarray
    sign: jnp.ndarray
    poly: jnp.ndarray
    coeff_1s: jnp.ndarray


class AtomicOrbital(NamedTuple):
    ind_nuc: int
    alpha: jnp.ndarray
    weights: jnp.ndarray
    angular_momenta: jnp.ndarray


class BaselineParams(NamedTuple):
    atomic_orbitals: AtomicOrbital
    cusp_params_up: CuspParams
    cusp_params_dn: CuspParams
    mo_coeff_up: jnp.ndarray
    mo_coeff_dn: jnp.ndarray
    ind_orb_up: jnp.ndarray
    ind_orb_dn: jnp.ndarray
    ci_weights: jnp.ndarray


class SystemState(NamedTuple):
    """State of the system"""

    hf_energy: float
    casscf_energy: float
    target_energy: jax.numpy.ndarray
    electron_spins: jax.numpy.ndarray
    nuclear_positions: jax.numpy.ndarray
    nuclear_charges: jax.numpy.ndarray
    n_el: jax.numpy.ndarray
    num_nuclei: int
    baseline_params: BaselineParams
    indices_u_u: jax.numpy.ndarray
    indices_d_d: jax.numpy.ndarray

    active_electrons: jax.numpy.ndarray
    active_nuclears: jax.numpy.ndarray
    n_up: int
    n_up_static: int
    n_el_static: int
    el_ion_mapping: jax.numpy.ndarray
    random_key: jax.random.PRNGKey = jax.random.PRNGKey(0)
    idx: int = 0
    ref_energy: float = 0.0


def zero_pad(x, n, pos="post"):
    if not isinstance(n, (list, tuple)):
        n = [n]
    if isinstance(pos, str):
        pos = [pos] * len(n)
    num_dims = len(n)
    padding = tuple(
        [
            (0, n[i] - x.shape[i]) if pos[i] == "post" else (n[i] - x.shape[i], 0)
            for i in range(num_dims)
        ]
    )
    return jnp.pad(x, padding, mode="constant")


def standartize_cusp_params(cusp_params: CuspParams, max_num_nuclei, max_num_orbitals):
    # jax.debug.breakpoint()
    new_rc = zero_pad(cusp_params.r_c, (max_num_nuclei))
    new_offset = zero_pad(cusp_params.offset, (max_num_nuclei, max_num_orbitals))
    new_sign = zero_pad(cusp_params.sign, (max_num_nuclei, max_num_orbitals))
    new_poly = zero_pad(cusp_params.poly, (max_num_nuclei, max_num_orbitals, 5))
    new_coeff_1s = zero_pad(
        cusp_params.coeff_1s, (max_num_nuclei, max_num_orbitals, max_num_orbitals)
    )
    return CuspParams(
        r_c=new_rc,
        offset=new_offset,
        sign=new_sign,
        poly=new_poly,
        coeff_1s=new_coeff_1s,
    )


def standartize_solution(
    baseline_params: BaselineParams,
    num_max_orbitals: int,
    num_max_electrons_up: int,
    num_max_electrons_dn: int,
    num_max_nuclei: int,
    num_determinants: int,
):
    assert (
        num_max_orbitals >= baseline_params.mo_coeff_up.shape[0]
    ), "num_max_orbitals should be greater than or equal to the number of orbitals in the baseline parameters"
    assert (
        num_max_electrons_up >= baseline_params.ind_orb_up.shape[1]
    ), "num_max_electrons should be greater than or equal to the number of electrons in the baseline parameters"

    MAX_K = 6
    # new_atomic_orbitals = [
    #     (ao[0], zero_pad(ao[1], MAX_K), zero_pad(ao[2], MAX_K), ao[3])
    #     for ao in baseline_params.atomic_orbitals
    # ]
    new_atomic_orbitals = [
        AtomicOrbital(
            ind_nuc=ao.ind_nuc,
            alpha=zero_pad(ao.alpha, MAX_K),
            weights=zero_pad(ao.weights, MAX_K),
            angular_momenta=ao.angular_momenta,
        )
        for ao in baseline_params.atomic_orbitals
    ]

    new_atomic_orbitals = new_atomic_orbitals + [new_atomic_orbitals[0]] * (
        num_max_orbitals - len(new_atomic_orbitals)
    )
    new_atomic_orbitals = jax.tree.map(lambda *x: jnp.stack(x), *new_atomic_orbitals)
    new_cusp_params_up = standartize_cusp_params(
        baseline_params.cusp_params_up, num_max_nuclei, num_max_orbitals
    )
    new_cusp_params_dn = standartize_cusp_params(
        baseline_params.cusp_params_dn, num_max_nuclei, num_max_orbitals
    )
    new_mo_coeff_up = zero_pad(
        baseline_params.mo_coeff_up, (num_max_orbitals, num_max_orbitals)
    )
    new_mo_coeff_dn = zero_pad(
        baseline_params.mo_coeff_dn, (num_max_orbitals, num_max_orbitals)
    )
    new_ind_orbitals_up = zero_pad(
        baseline_params.ind_orb_up, (num_determinants, num_max_electrons_up)
    )
    new_ind_orbitals_dn = zero_pad(
        baseline_params.ind_orb_dn, (num_determinants, num_max_electrons_dn)
    )

    new_ci_weights = zero_pad(baseline_params.ci_weights, (num_determinants,))
    return BaselineParams(
        atomic_orbitals=new_atomic_orbitals,
        cusp_params_up=new_cusp_params_up,
        cusp_params_dn=new_cusp_params_dn,
        mo_coeff_up=new_mo_coeff_up,
        mo_coeff_dn=new_mo_coeff_dn,
        ind_orb_up=new_ind_orbitals_up,
        ind_orb_dn=new_ind_orbitals_dn,
        ci_weights=new_ci_weights,
    )
