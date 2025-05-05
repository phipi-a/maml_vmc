import json
import shutil
from typing import List

from maml_vmc.sampler.utils import (
    SystemState,
    standartize_solution,
    zero_pad,
)
import pickle
import os
import numpy as np
from maml_vmc.models.deep_erwin.orbitals import get_baseline_solution
import jax.numpy as jnp


def build_specific_config(
    R, Z, el_ion_mapping, n_electrons, n_orbitals, n_up, n_dets, ref_energy=0.0
):
    n_nuclei = len(Z)
    n_cas_orbitals = n_orbitals
    n_cas_electrons = n_electrons
    n_determinants = n_dets
    n_dn = n_electrons - n_up

    bl_params, hf_energy, casscf_energy = get_baseline_solution(
        R, Z, n_electrons, n_up, n_cas_orbitals, n_cas_electrons, n_determinants
    )

    indices_u_u = np.array(
        [[j for j in range(n_up) if j != i] for i in range(n_up)], dtype=int
    )
    # indices for the other downspinning electrons regarding each downspinning electron
    indices_d_d = np.array(
        [[j + n_up for j in range(n_dn) if j != i] for i in range(n_dn)], dtype=int
    )
    return SystemState(
        hf_energy=hf_energy,
        casscf_energy=casscf_energy,
        target_energy=0.0,
        electron_spins=jnp.array([1] * n_up + [-1] * (n_electrons - n_up)),
        nuclear_positions=R,
        nuclear_charges=Z,
        n_el=n_electrons,
        baseline_params=bl_params,
        active_electrons=jnp.ones((n_electrons)),
        active_nuclears=jnp.ones_like(Z),
        el_ion_mapping=jnp.array(el_ion_mapping),
        num_nuclei=n_nuclei,
        indices_u_u=jnp.array(indices_u_u),
        indices_d_d=jnp.array(indices_d_d),
        n_up=n_up,
        n_up_static=n_up,
        n_el_static=n_electrons,
        ref_energy=ref_energy,
    )


def standartize_configs(
    configs: List[SystemState], n_dets, max_num_electrons=None, max_num_nuclei=None
):
    def get_max_sizes(configs: List[SystemState]):
        max_orbitals = max(
            [config.baseline_params.mo_coeff_up.shape[0] for config in configs]
        )
        return max_orbitals

    keep_sizes = False
    if max_num_electrons is None or max_num_nuclei is None:
        keep_sizes = True

    max_orbitals = get_max_sizes(configs)
    new_configs = []
    for config in configs:
        indices_u_u = config.indices_u_u
        indices_d_d = config.indices_d_d
        if keep_sizes:
            max_num_electrons = config.n_el
            max_num_nuclei = config.num_nuclei
        else:
            indices_u_u = jnp.zeros((max_num_electrons, max_num_electrons))
            indices_d_d = jnp.zeros((max_num_electrons, max_num_electrons))
        new_casscf_solution = standartize_solution(
            config.baseline_params,
            max_orbitals,
            config.n_up if keep_sizes else max_num_electrons,
            config.n_el - config.n_up if keep_sizes else max_num_electrons,
            max_num_nuclei,
            n_dets,
        )

        new_config = SystemState(
            hf_energy=config.hf_energy,
            casscf_energy=config.casscf_energy,
            target_energy=config.target_energy,
            electron_spins=zero_pad(config.electron_spins, max_num_electrons).astype(
                jnp.int32
            ),
            nuclear_positions=zero_pad(config.nuclear_positions, (max_num_nuclei, 3)),
            nuclear_charges=zero_pad(config.nuclear_charges, max_num_nuclei).astype(
                jnp.int32
            ),
            el_ion_mapping=zero_pad(config.el_ion_mapping, max_num_electrons).astype(
                jnp.int32
            ),
            n_el=config.n_el,
            num_nuclei=config.num_nuclei,
            baseline_params=new_casscf_solution,
            active_electrons=zero_pad(
                config.active_electrons, max_num_electrons
            ).astype(jnp.bool_),
            active_nuclears=zero_pad(config.active_nuclears, max_num_nuclei).astype(
                jnp.bool_
            ),
            indices_u_u=indices_u_u,
            indices_d_d=indices_d_d,
            n_up=config.n_up,
            n_up_static=config.n_up_static if keep_sizes else 0,
            n_el_static=config.n_el_static if keep_sizes else 0,
            ref_energy=config.ref_energy,
        )
        new_configs.append(new_config)
    if keep_sizes:
        max_num_electrons = -1
        max_num_nuclei = -1
    return new_configs, max_orbitals, max_num_electrons, max_num_nuclei


def store_configs(
    configs: List[SystemState],
    num_orbitals,
    num_electrons,
    num_nuclei,
    path,
):
    path = os.path.join(path, "configs")
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

    for i, config in enumerate(configs):
        with open(
            os.path.join(
                path,
                f"config_n{config.num_nuclei}_el{config.n_el}_i{i}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(config, f)

    with open(os.path.join(path, "max_sizes.json"), "w") as f:
        json.dump(
            {
                "max_orbitals": num_orbitals,
                "max_electrons": num_electrons,
                "max_nuclei": num_nuclei,
            },
            f,
        )
