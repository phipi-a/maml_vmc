"""
Contains the physical baseline model (e.g. CASSCF).

This module provides functionality to calculate a basline solution using pySCF and functions to
"""

import math
import jax
import jax.numpy as jnp

import numpy as np

np.random.seed(0)
import pyscf
import pyscf.md


import pyscf
import pyscf.mcscf

from maml_vmc.sampler.MoleculeDataSampler import BaselineParams
from maml_vmc.sampler.utils import AtomicOrbital, CuspParams

pyscf.md.set_seed(0)
np.random.seed(0)

#################################################################################
############################ Orbital functions ##################################
#################################################################################


import jax.numpy as jnp


def eval_gaussian_orbital(el_ion_diff, el_ion_dist, ao):
    """
    Evaluate a single Gaussian orbital at multiple points in space.
    """
    r_sqr = jnp.expand_dims((el_ion_dist**2), axis=-1)
    el_ion_dist = jnp.where(el_ion_dist == 0, 1e-12, el_ion_dist)
    pre_fac = jnp.prod(el_ion_diff**ao.angular_momenta, axis=-1)

    max_exp = jnp.max(-ao.alpha * r_sqr)
    stable_exp = jnp.exp(-ao.alpha * r_sqr - max_exp)

    phi_gto = pre_fac * jnp.sum(ao.weights * stable_exp, axis=-1) * jnp.exp(max_exp)

    return phi_gto


def eval_atomic_orbitals(el_ion_diff, el_ion_dist, orbitals: AtomicOrbital):
    """
    Args:
        el_ion_diff: shape [N_batch x n_el x N_ion x 3]
        el_ion_dist: shape [N_batch x n_el x N_ion]
        orbitals:

    Returns:

    """
    if isinstance(orbitals, list):
        outputs = []
        for i, ao in enumerate(orbitals):
            ind_nuc = ao[0]
            diff = el_ion_diff[..., ind_nuc, :]
            dist = el_ion_dist[..., ind_nuc]
            gto = eval_gaussian_orbital(diff, dist, ao)
            outputs.append(gto)
        return jnp.stack(outputs, axis=-1)

    def eval_orbital(ao: AtomicOrbital):
        ind_nuc = ao.ind_nuc
        diff = el_ion_diff[..., ind_nuc, :]
        dist = el_ion_dist[..., ind_nuc]
        gto = eval_gaussian_orbital(diff, dist, ao)
        return gto

    out = jax.vmap(eval_orbital, in_axes=(0), out_axes=-1)(orbitals)

    return out


def _eval_cusp_atomic_orbital(dist, r_c, offset, sign, poly):
    dist = jnp.minimum(dist, r_c)
    n = jnp.arange(5)
    r_n = dist[..., jnp.newaxis] ** n
    p_r = jnp.dot(r_n, poly)
    psi = offset + sign * jnp.exp(p_r)
    return psi


def _eval_cusp_molecular_orbital(dist, r_c, offset, sign, poly):
    dist = jnp.minimum(dist, r_c)
    # dist for dummy atoms is 0, so we need to add a small value to avoid division by zero
    eps = 1e-12
    dist = jnp.where(dist == 0, eps, dist)
    # el_ion_dist
    gn = jnp.arange(1, 5)

    r_n = dist[..., jnp.newaxis, jnp.newaxis] ** gn
    r_no = jnp.ones_like(dist)[..., jnp.newaxis, jnp.newaxis]
    r_n = jnp.concatenate([r_no, r_n], axis=-1)
    p_r = jnp.sum(r_n * poly, axis=-1)
    psi = offset + sign * jnp.exp(p_r)
    return psi


def evaluate_molecular_orbitals(
    el_ion_diff,
    el_ion_dist,
    atomic_orbitals: AtomicOrbital,
    mo_coeff,
    cusp_params: CuspParams,
    electron_mask,
):
    aos = eval_atomic_orbitals(el_ion_diff, el_ion_dist, atomic_orbitals)

    mos = aos @ mo_coeff

    mos = mos * electron_mask[:, None]

    # cusp correction
    # sto_mask: rc > el_ion_dist
    sto_mask = jnp.heaviside(cusp_params.r_c[None, :] - el_ion_dist, 0.0)[
        ..., jnp.newaxis
    ]
    cusp_mo = _eval_cusp_molecular_orbital(
        el_ion_dist,
        cusp_params.r_c,
        cusp_params.offset,
        cusp_params.sign,
        cusp_params.poly,
    )
    sto = cusp_mo - jnp.sum(
        aos[..., np.newaxis, :, np.newaxis] * cusp_params.coeff_1s, axis=-2
    )
    sto = jnp.sum(
        sto * sto_mask, axis=-2
    )  # sum over all ions, masking only the once that are within r_c of an electron
    sto = sto * electron_mask[:, None]
    mos = mos + sto  # subtract GTO-part of 1s orbitals and replace by STO-part
    return mos


#################################################################################
####################### Compile-time helper functions ###########################
#################################################################################


def build_pyscf_molecule(R, Z, charge=0, spin=0, basis_set="6-311G"):
    """
    Args:
        R: ion positions, shape [Nx3]
        Z: ion charges, shape [N]
        charge: integer
        spin: integer; number of excess spin-up electrons
        basis_set (str): basis set identifier

    Returns:
        pyscf.Molecule
    """
    molecule = pyscf.gto.Mole()
    molecule.atom = [[Z_, tuple(R_)] for R_, Z_ in zip(R, Z)]

    molecule.unit = "bohr"
    molecule.basis = basis_set
    molecule.cart = True
    molecule.spin = spin  # 2*n_up - n_down
    molecule.charge = charge
    molecule.output = "/dev/null"
    molecule.verbose = 0  # suppress output to console
    molecule.max_memory = 10e3  # maximum memory in megabytes (i.e. 10e3 = 10GB)
    molecule.build()
    return molecule


def _get_gto_normalization_factor(alpha, angular_momenta):
    l_tot = np.sum(angular_momenta)
    fac_alpha = (2 * alpha / np.pi) ** (3 / 4) * (8 * alpha) ** (l_tot / 2)

    fac = np.array([math.factorial(x) for x in angular_momenta])
    fac2 = np.array([math.factorial(2 * x) for x in angular_momenta])
    fac_exponent = np.sqrt(np.prod(fac) / np.prod(fac2))
    factor = fac_alpha * fac_exponent
    return factor


def _get_atomic_orbital_basis_functions(molecule):
    """
    Args:
        molecule: pyscf molecule object
    Returns:
        orbitals: tuple of tuple, each containing index of nucleus, nuclear coordinate, exponential decay constant alpha, weights and angular momentum exponents
    """
    ao_labels = molecule.ao_labels(None)
    n_basis = len(ao_labels)
    ind_basis = 0
    atomic_orbitals = []
    for ind_nuc, (element, atom_pos) in enumerate(molecule._atom):
        for gto_data in molecule._basis[element]:
            l = gto_data[0]
            gto_data = np.array(gto_data[1:])
            alpha = gto_data[:, 0]
            weights = gto_data[:, 1:]
            for ind_contraction in range(weights.shape[1]):
                for m in range(-l, l + 1):
                    shape_string = ao_labels[ind_basis][
                        3
                    ]  # string of the form 'xxy' or 'zz'
                    angular_momenta = np.array(
                        [shape_string.count(x) for x in ["x", "y", "z"]], dtype=int
                    )
                    normalization = _get_gto_normalization_factor(
                        alpha, angular_momenta
                    )
                    # atomic_orbitals.append(
                    #     [
                    #         ind_nuc,
                    #         alpha,
                    #         weights[:, ind_contraction] * normalization,
                    #         angular_momenta,
                    #     ]
                    # )
                    atomic_orbitals.append(
                        AtomicOrbital(
                            ind_nuc=ind_nuc,
                            alpha=alpha,
                            weights=weights[:, ind_contraction] * normalization,
                            angular_momenta=angular_momenta,
                        )
                    )
                    ind_basis += 1
    assert (
        len(atomic_orbitals) == n_basis
    ), "Could not properly construct basis functions. You probably tried to use a valence-only basis (e.g. cc-pVDZ) for an all-electron calculation."  # noqa
    return atomic_orbitals


def get_baseline_solution(
    R,
    Z,
    n_electrons,
    n_up,
    n_cas_orbitals,
    n_cas_electrons,
    n_determinants,
    r_cusp_el_ion_scale=1.0,
):

    charge = sum(Z) - n_electrons
    spin = 2 * n_up - n_electrons
    basis_set = "6-311G"
    molecule = build_pyscf_molecule(R, Z, charge, spin, basis_set)
    atomic_orbitals = _get_atomic_orbital_basis_functions(molecule)

    hf = pyscf.scf.HF(molecule)
    hf.verbose = 0  # suppress output to console
    hf.kernel()

    casscf = pyscf.mcscf.UCASSCF(hf, n_cas_orbitals, n_cas_electrons)
    casscf.kernel()

    mo_coeff = list(casscf.mo_coeff)  # tuple of spin_up, spin_down
    ind_orbitals = _get_orbital_indices(casscf)
    ci_weights = casscf.ci.flatten()
    if n_determinants < len(ci_weights):
        ind_largest = np.argsort(np.abs(ci_weights))[::-1][:n_determinants]
        share_captured = np.sum(ci_weights[ind_largest] ** 2) / np.sum(ci_weights**2)

        ci_weights = ci_weights[ind_largest]
        ci_weights = jnp.array(ci_weights / np.sum(ci_weights**2))
        ind_orbitals = ind_orbitals[0][ind_largest], ind_orbitals[1][ind_largest]

    # delete unused molecular orbitals (i.e. orbitals that appear in none of the selected determinants)
    # for spin in range(2):
    #     n_mo_max = np.max(ind_orbitals[spin]) + 1
    #     mo_coeff[spin] = mo_coeff[spin][:, :n_mo_max]

    # Calculate cusp-correction-parameters for molecular orbitals
    cusp_params = [
        calculate_molecular_orbital_cusp_params(
            atomic_orbitals,
            mo_coeff[i],
            R,
            Z,
            r_cusp_el_ion_scale,
        )
        for i in range(2)
    ]

    bp = BaselineParams(
        atomic_orbitals=atomic_orbitals,
        cusp_params_up=cusp_params[0],
        cusp_params_dn=cusp_params[1],
        mo_coeff_up=mo_coeff[0],
        mo_coeff_dn=mo_coeff[1],
        ind_orb_up=ind_orbitals[0],
        ind_orb_dn=ind_orbitals[1],
        ci_weights=ci_weights,
    )

    return (
        bp,
        hf.e_tot,
        casscf.e_tot,
    )


def _int_to_spin_tuple(x):
    if type(x) == int:
        return (x,) * 2
    else:
        return x


def _get_orbitals_by_cas_type(casscf):
    """
    Splits orbitals into fixed orbitals (that are either always occupied, or always unoccupied) and active orbitals (that are occupied in some determinants, and unoccupied in others).

    Returns:
        tuple containing

        - **fixed_orbitals** (list of 2 lists): For each spin, a list of indices of fixed orbitals
        - **active_orbitals** (list of 2 lists): For each spin, a list of indices of active orbitals
    """
    n_core = _int_to_spin_tuple(casscf.ncore)
    n_cas = _int_to_spin_tuple(casscf.ncas)

    active_orbitals = [list(range(n_core[s], n_core[s] + n_cas[s])) for s in range(2)]
    fixed_orbitals = [list(range(n_core[s])) for s in range(2)]
    return fixed_orbitals, active_orbitals


def _get_orbital_indices(casscf):
    """
    Parse the output of the pySCF CASSCF calculation to determine which orbitals are included in which determinant.

    Returns:
        (list of 2 np.arrays): First array for spin-up, second array for spin-down. Each array has shape [N_determinants x n_electrons_of_spin] and contains the indices of occupied orbitals in each determinant.
    """
    fixed, active = _get_orbitals_by_cas_type(casscf)

    nelcas = _int_to_spin_tuple(casscf.nelecas)
    occ_up = pyscf.fci.cistring._gen_occslst(active[0], nelcas[0])
    occ_down = pyscf.fci.cistring._gen_occslst(active[1], nelcas[1])

    orbitals_up = []
    orbitals_dn = []
    for o_up in occ_up:
        for o_dn in occ_down:
            orbitals_up.append(fixed[0] + list(o_up))
            orbitals_dn.append(fixed[1] + list(o_dn))
    return [jnp.array(orbitals_up, dtype=int), jnp.array(orbitals_dn, dtype=int)]


##########################################################################################
##################################### Cusp functions #####################################
##########################################################################################


def build_el_el_cusp_correction(n_electrons, n_up, config):
    factor = np.ones([n_electrons, n_electrons - 1]) * 0.5
    factor[:n_up, : n_up - 1] = 0.25
    factor[n_up:, n_up:] = 0.25
    A = config.r_cusp_el_el**2
    B = config.r_cusp_el_el

    def el_el_cusp(el_el_dist):
        # No factor 0.5 here, e.g. when comparing to NatChem 2020, [doi.org/10.1038/s41557-020-0544-y], because:
        # A) We double-count electron-pairs because we take the full distance matrix (and not only the upper triangle)
        # B) We model log(psi^2)=2*log(|psi|) vs log(|psi|) int NatChem 2020, i.e. the cusp correction needs a factor 2
        return -jnp.sum(factor * A / (el_el_dist + B), axis=[-2, -1])

    return el_el_cusp


def _get_local_energy_of_cusp_orbital(r_c, offset, sign, poly, Z, phi_others=0.0):
    r = np.linspace(1e-6, r_c, 100)[:, np.newaxis]
    p_0 = poly[0] + poly[1] * r + poly[2] * r**2 + poly[3] * r**3 + poly[4] * r**4
    p_1 = poly[1] + 2 * poly[2] * r + 3 * poly[3] * r**2 + 4 * poly[4] * r**3
    p_2 = 2 * poly[2] + 6 * poly[3] * r + 12 * poly[4] * r**2
    prefac = sign * np.exp(p_0) / (offset + sign * np.exp(p_0) + phi_others)
    E_l = -prefac * (p_1 / r + 0.5 * p_2 + 0.5 * p_1**2) - Z / r
    penalty = np.nanvar(E_l, axis=0)
    # penalty = jnp.max(jnp.abs(E_l - E_l[-1]), axis=0)
    return penalty


def _calculate_mo_cusp_params(
    phi_rc_0, phi_rc_1, phi_rc_2, phi_0, phi_0_others, Z, r_c
):
    if np.abs(phi_0) < 1e-6 and np.abs(phi_rc_0) < 1e-6:
        return 0.0, 0.0, np.zeros(5)
    n_cusp_trials = 500
    phi_new_0 = phi_0 * (
        1.0
        + np.concatenate(
            [
                np.logspace(-2, 1, n_cusp_trials // 2),
                -np.logspace(-2, 1, n_cusp_trials // 2),
            ]
        )
    )

    sign = jnp.sign(phi_new_0 - phi_rc_0)
    offset = 2 * phi_rc_0 - phi_new_0  # = "C"
    phi_rc_shifted = phi_rc_0 - offset  # = "R(r_c)"

    x1 = np.log(jnp.abs(phi_rc_shifted))
    x2 = phi_rc_1 / phi_rc_shifted
    x3 = phi_rc_2 / phi_rc_shifted
    x4 = -Z * (phi_new_0 + phi_0_others) / (phi_new_0 - offset)
    x5 = np.log(np.abs(phi_new_0 - offset))

    rc2 = r_c * r_c
    rc3 = rc2 * r_c
    rc4 = rc2 * rc2
    poly = np.array(
        [
            x5,
            x4,
            6 * x1 / rc2
            - 3 * x2 / r_c
            + 0.5 * x3
            - 3 * x4 / r_c
            - 6 * x5 / rc2
            - 0.5 * x2 * x2,
            -8 * x1 / rc3
            + 5 * x2 / rc2
            - x3 / r_c
            + 3 * x4 / rc2
            + 8 * x5 / rc3
            + x2 * x2 / r_c,
            3 * x1 / rc4
            - 2 * x2 / rc3
            + 0.5 * x3 / rc2
            - x4 / rc3
            - 3 * x5 / rc4
            - 0.5 * x2 * x2 / rc2,
        ]
    )
    E_loc_cusp = _get_local_energy_of_cusp_orbital(
        r_c, offset, sign, poly, Z, phi_0_others
    )
    ind_opt = np.nanargmin(E_loc_cusp)
    return offset[ind_opt], sign[ind_opt], poly[:, ind_opt]


def _calculate_ao_cusp_params(ind_nuc, alpha, gto_coeffs, angular_momenta, Z):
    if sum(angular_momenta) != 0:
        return None
    r_c = jnp.minimum(0.5, 1 / Z)
    phi_rc_0 = jnp.sum(gto_coeffs * np.exp(-alpha * r_c**2))
    phi_rc_1 = jnp.sum(gto_coeffs * (-2 * alpha * r_c) * np.exp(-alpha * r_c**2))
    phi_rc_2 = jnp.sum(
        gto_coeffs * (-2 * alpha + 4 * (alpha * r_c) ** 2) * np.exp(-alpha * r_c**2)
    )
    phi_0 = jnp.sum(gto_coeffs)

    n_cusp_trials = 500
    phi_new_0 = phi_0 * np.linspace(-1, 3, n_cusp_trials)

    sign = jnp.sign(phi_new_0 - phi_rc_0)
    offset = 2 * phi_rc_0 - phi_new_0
    phi_shifted = phi_rc_0 - offset

    p0 = jnp.log((phi_new_0 - offset) * sign)
    p1 = -Z * (offset * sign * jnp.exp(-p0) + 1)

    A = jnp.array(
        [
            [r_c**2, r_c**3, r_c**4],
            [2 * r_c, 3 * r_c**2, 4 * r_c**3],
            [2, 6 * r_c, 12 * r_c**2],
        ]
    )
    b = jnp.array(
        [
            jnp.log(phi_shifted * sign) - p0 - p1 * r_c,
            phi_rc_1 / phi_shifted - p1,
            phi_rc_2 / phi_shifted - (phi_rc_1 / phi_shifted) ** 2,
        ]
    )
    poly = jnp.concatenate([jnp.array([p0, p1]), jnp.linalg.solve(A, b)])
    E_loc_cusp = _get_local_energy_of_cusp_orbital(
        r_c, offset, sign, poly, Z, phi_others=0.0
    )
    ind_opt = jnp.nanargmin(E_loc_cusp)
    return r_c, offset[ind_opt], sign[ind_opt], poly[:, ind_opt]


def get_el_ion_distance_matrix(r_el, R_ion):
    """
    Args:
        r_el: shape [N_batch x n_el x 3]
        R_ion: shape [N_ion x 3]
    Returns:
        diff: shape [N_batch x n_el x N_ion x 3]
        dist: shape [N_batch x n_el x N_ion]
    """
    diff = jnp.expand_dims(r_el, -2) - R_ion
    dist = jnp.linalg.norm(diff, axis=-1)
    return diff, dist


def calculate_molecular_orbital_cusp_params(
    atomic_orbitals, mo_coeff, R, Z, r_cusp_scale
):
    n_molecular_orbitals, n_nuclei, n_atomic_orbitals = (
        mo_coeff.shape[1],
        len(R),
        len(atomic_orbitals),
    )
    cusp_rc = jnp.minimum(r_cusp_scale / Z, 0.5)
    cusp_offset = np.zeros([n_nuclei, n_molecular_orbitals])
    cusp_sign = np.zeros([n_nuclei, n_molecular_orbitals])
    cusp_poly = np.zeros([n_nuclei, n_molecular_orbitals, 5])
    cusp_1s_coeffs = np.zeros([n_nuclei, n_atomic_orbitals, n_molecular_orbitals])
    for nuc_idx in range(n_nuclei):
        for mo_idx in range(n_molecular_orbitals):
            diff, dist = get_el_ion_distance_matrix(jnp.array(R[nuc_idx]), R)
            ao = eval_atomic_orbitals(diff, dist, atomic_orbitals)
            is_centered_1s = np.array(
                [(a[0] == nuc_idx) and (sum(a[3]) == 0) for a in atomic_orbitals]
            )
            phi_0_1s = (is_centered_1s * ao) @ mo_coeff[:, mo_idx]
            phi_0_others = ((1 - is_centered_1s) * ao) @ mo_coeff[:, mo_idx]
            phi_rc_0, phi_rc_1, phi_rc_2 = 0.0, 0.0, 0.0
            r_c = cusp_rc[nuc_idx]
            for i, (_, alpha, weights, _) in enumerate(atomic_orbitals):
                if is_centered_1s[i]:
                    phi_rc_0 += (
                        jnp.sum(weights * np.exp(-alpha * r_c**2)) * mo_coeff[i, mo_idx]
                    )
                    phi_rc_1 += (
                        jnp.sum(weights * (-2 * alpha * r_c) * np.exp(-alpha * r_c**2))
                        * mo_coeff[i, mo_idx]
                    )
                    phi_rc_2 += (
                        jnp.sum(
                            weights
                            * (-2 * alpha + 4 * (alpha * r_c) ** 2)
                            * np.exp(-alpha * r_c**2)
                        )
                        * mo_coeff[i, mo_idx]
                    )
            cusp_1s_coeffs[nuc_idx, :, mo_idx] = (
                is_centered_1s * mo_coeff[:, mo_idx]
            )  # n_nuc x n_atomic_orbitals x n_molec_orbitals
            (
                cusp_offset[nuc_idx, mo_idx],
                cusp_sign[nuc_idx, mo_idx],
                cusp_poly[nuc_idx, mo_idx],
            ) = _calculate_mo_cusp_params(
                phi_rc_0, phi_rc_1, phi_rc_2, phi_0_1s, phi_0_others, Z[nuc_idx], r_c
            )
    return CuspParams(cusp_rc, cusp_offset, cusp_sign, cusp_poly, cusp_1s_coeffs)
    # return cusp_rc, cusp_offset, cusp_sign, cusp_poly, cusp_1s_coeffs
