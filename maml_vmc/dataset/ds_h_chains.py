from maml_vmc.dataset.utils import (
    build_specific_config,
    standartize_configs,
    store_configs,
)
import numpy as np
from tqdm import tqdm
import jsonargparse
import json


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str)
    parser.add_argument("--output", "-o", type=str)
    n_det = 32
    args = parser.parse_args()
    in_dict = json.load(open(args.input))

    bar = tqdm(total=len(in_dict))
    configs = []
    max_chain_length = 0
    for i, cdef in enumerate(in_dict):

        i = int(i)
        R = np.array(cdef["R"])
        n_electrons = cdef["num_electrons"]

        if R.shape[0] > max_chain_length:
            max_chain_length = R.shape[0]
        Z = np.array(cdef["atom_charges"])
        n_orbitals = n_electrons
        n_up = n_electrons - (n_electrons // 2)
        up_el_ion_mapping = np.array([i * 2 for i in range(n_up)])
        down_el_ion_mapping = np.array([i * 2 + 1 for i in range(n_electrons - n_up)])
        el_ion_mapping = np.concatenate([up_el_ion_mapping, down_el_ion_mapping])
        c = build_specific_config(
            R,
            Z,
            el_ion_mapping,
            n_electrons,
            n_orbitals,
            n_up,
            n_det,
            ref_energy=cdef["ref_energy"],
        )
        configs.append(c)

        bar.update(1)
    bar.close

    new_configs, max_num_orbitals, max_num_electrons, max_num_nuclei = (
        standartize_configs(configs, n_det, max_chain_length, max_chain_length)
    )
    store_configs(
        new_configs,
        max_num_orbitals,
        max_num_electrons,
        max_num_nuclei,
        args.output,
    )
