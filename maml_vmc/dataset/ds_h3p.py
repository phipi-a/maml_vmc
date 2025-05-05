from maml_vmc.dataset.utils import (
    build_specific_config,
    standartize_configs,
    store_configs,
)
import numpy as np
from tqdm import tqdm
import jsonargparse
import json
from pyscf import lib

lib.num_threads(1)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str)
    parser.add_argument("--output", "-o", type=str)
    n_det = 20
    args = parser.parse_args()
    in_dict = json.load(open(args.input))

    bar = tqdm(total=len(in_dict))
    configs = []
    for i, cdef in enumerate(in_dict):

        i = int(i)
        R = np.array(cdef["R"])
        n_electrons = cdef["num_electrons"]
        Z = np.array(cdef["atom_charges"])
        n_orbitals = 6
        n_up = 1
        el_ion_mapping = [0, 1]
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
        standartize_configs(configs, n_det, 2, 3)
    )

    store_configs(
        new_configs,
        max_num_orbitals,
        max_num_electrons,
        max_num_nuclei,
        args.output,
    )
