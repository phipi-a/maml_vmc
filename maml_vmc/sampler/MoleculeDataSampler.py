from dataclasses import dataclass
import functools
import json
import os
import pickle
import random
from typing import NamedTuple
import jax
from maml_vmc import lijax
import jax.numpy as jnp
import haiku as hk

from maml_vmc.sampler import MetropolisSampler
from maml_vmc.sampler.utils import BaselineParams, SystemState


class MoleculeDataSamplerSelfState(NamedTuple):
    num_burnin_steps: int
    num_samples: int
    num_steps: int
    num_inter_burnin_steps: int
    step_size: float
    dataset_path: str
    use_variable_size: bool = False
    file_name: str = None
    add_noise: bool = False

    def __hash__(self):
        return hash(
            (
                self.num_burnin_steps,
                self.num_samples,
                self.num_steps,
                self.num_inter_burnin_steps,
                self.step_size,
                self.dataset_path,
                self.use_variable_size,
                self.file_name,
                self.add_noise,
            )
        )


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "electron_positions",
        "electron_spins",
        "nuclear_positions",
        "nuclear_charges",
        "baseline_params",
        "random_key",
        "active_nuclears",
        "active_electrons",
        "n_up",
        "n_el",
        "n_dn",
        "indices_u_u",
        "indices_d_d",
    ],
    meta_fields=["n_up_static", "n_dn_static", "n_el_static"],
)
@dataclass
class MoleculeFeatures:
    """Features of the system"""

    electron_positions: jnp.ndarray
    electron_spins: jnp.ndarray
    nuclear_positions: jnp.ndarray
    nuclear_charges: jnp.ndarray
    baseline_params: BaselineParams
    random_key: jnp.ndarray
    active_nuclears: jnp.ndarray
    active_electrons: jnp.ndarray
    n_up: int
    n_dn: int
    n_el: int
    indices_u_u: jnp.ndarray
    indices_d_d: jnp.ndarray
    n_up_static: int = 0
    n_dn_static: int = 0
    n_el_static: int = 0


class MoleculeDataSampler(lijax.DataSampler):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

        self_state = MoleculeDataSamplerSelfState(*args, **kwargs)

        list_of_files = os.listdir(self_state.dataset_path)
        list_of_files = [f for f in list_of_files if f.endswith(".pkl")]
        if self_state.file_name is not None:
            list_of_files = [self_state.file_name]
        # randomize list of files
        # shuffle list by sorting by random number
        random.seed(0)
        random.shuffle(list_of_files)
        self.list_of_files = list_of_files
        print(self.list_of_files)
        configs = [
            pickle.load(open(os.path.join(self_state.dataset_path, f), "rb"))
            for f in list_of_files
        ]
        # set type of configs
        self.sample_config = configs[0]

        self.configs: SystemState = jax.tree.map(lambda *x: jnp.stack(x), *configs)
        max_sizes = json.load(
            open(os.path.join(self_state.dataset_path, "max_sizes.json"), "r")
        )
        self.num_max_orbitals = max_sizes["max_orbitals"]
        self.num_max_electrons = max_sizes["max_electrons"]
        self.num_max_nuclei = max_sizes["max_nuclei"]
        self.num_systems = len(configs)

        self.self_state = self_state

    def get_fake_datapoint(self):
        self.sample_config: SystemState = self.sample_config
        return MoleculeFeatures(
            electron_positions=jax.random.normal(
                jax.random.PRNGKey(0),
                (
                    (
                        self.sample_config.n_el
                        if self.self_state.use_variable_size
                        else self.num_max_electrons
                    )
                    * 3,
                ),
            ),
            electron_spins=self.sample_config.electron_spins,
            nuclear_positions=self.sample_config.nuclear_positions,
            nuclear_charges=self.sample_config.nuclear_charges,
            baseline_params=self.sample_config.baseline_params,
            random_key=jax.random.PRNGKey(0),
            active_nuclears=self.sample_config.active_nuclears,
            active_electrons=self.sample_config.active_electrons,
            n_up=self.sample_config.n_up,
            n_dn=self.sample_config.n_el - self.sample_config.n_up,
            n_el=self.sample_config.n_el,
            indices_u_u=self.sample_config.indices_u_u,
            indices_d_d=self.sample_config.indices_d_d,
        )

    def sample_system(self, prng_key: jax.random.PRNGKey, idx: int = -1):
        idx_random = jax.random.randint(prng_key, (1,), 0, self.num_systems)[0]
        idx = jnp.where(idx == -1, idx_random, idx)
        cf = jax.tree.map(lambda x: x[idx], self.configs)
        cf_dict = cf._asdict()
        cf_dict["random_key"] = jax.random.PRNGKey(0)
        cf_dict["idx"] = idx
        if self.self_state.add_noise:
            cf_dict["nuclear_positions"] = cf_dict["nuclear_positions"] * (
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (1, 1),
                )
                * 0.1
                + 1.0
            )
        ss = SystemState(**cf_dict)

        return ss

    def sample_data(
        self,
        prng_key: jax.random.PRNGKey,
        system_state: SystemState,
        model: hk.Transformed,
        model_params: hk.Params,
        init_walker_state: MetropolisSampler.MetropolisWalkerState = None,
    ):

        def eval_fn(x_pos, x_spin):
            features = MoleculeFeatures(
                electron_positions=x_pos.astype(jnp.float32),
                electron_spins=system_state.electron_spins,
                nuclear_positions=system_state.nuclear_positions,
                nuclear_charges=system_state.nuclear_charges,
                random_key=system_state.random_key,
                baseline_params=system_state.baseline_params,
                n_up=system_state.n_up,
                n_el=system_state.n_el,
                n_dn=system_state.n_el - system_state.n_up,
                indices_u_u=system_state.indices_u_u,
                indices_d_d=system_state.indices_d_d,
                active_electrons=system_state.active_electrons,
                active_nuclears=system_state.active_nuclears,
            )
            out = model.apply(model_params, features)
            return out

        num_burn_in = (
            self.self_state.num_burnin_steps
            if init_walker_state is None
            else self.self_state.num_inter_burnin_steps
        )

        ms, ms_last_state = MetropolisSampler.run_metropolis_batch(
            prng_key,
            self.self_state.num_samples,
            self.num_max_electrons * 3,
            0,
            num_burn_in,
            self.self_state.num_steps,
            self.self_state.step_size,
            system_state.el_ion_mapping,
            system_state.nuclear_positions,
            eval_fn,
            init_walker_state=init_walker_state,
        )
        # combine the metropolis steps and walkers
        batch = jax.tree.map(
            lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]), ms
        )

        feature_batch = MoleculeFeatures(
            electron_positions=batch.position,
            nuclear_positions=system_state.nuclear_positions[None, :, :].repeat(
                batch.position.shape[0], axis=0
            ),
            nuclear_charges=system_state.nuclear_charges[None, :].repeat(
                batch.position.shape[0], axis=0
            ),
            n_el=system_state.n_el[None].repeat(batch.position.shape[0], axis=0),
            random_key=system_state.random_key[None].repeat(
                batch.position.shape[0], axis=0
            ),
            electron_spins=system_state.electron_spins[None, :].repeat(
                batch.position.shape[0], axis=0
            ),
            baseline_params=jax.tree.map(
                lambda x: x[None].repeat(batch.position.shape[0], axis=0),
                system_state.baseline_params,
            ),
            n_up=system_state.n_up[None].repeat(batch.position.shape[0], axis=0),
            n_dn=(system_state.n_el - system_state.n_up)[None].repeat(
                batch.position.shape[0], axis=0
            ),
            indices_u_u=system_state.indices_u_u[None].repeat(
                batch.position.shape[0], axis=0
            ),
            indices_d_d=system_state.indices_d_d[None].repeat(
                batch.position.shape[0], axis=0
            ),
            active_electrons=system_state.active_electrons[None].repeat(
                batch.position.shape[0], axis=0
            ),
            active_nuclears=system_state.active_nuclears[None].repeat(
                batch.position.shape[0], axis=0
            ),
        )
        return (feature_batch, ms_last_state)

    @staticmethod
    # @jax.jit
    def potential_fn(mol: MoleculeFeatures, system_state: SystemState):
        el_pos = mol.electron_positions.reshape(-1, 3)
        r_ea_diff = el_pos[:, None, :] - mol.nuclear_positions[None, :, :]

        r_ea_diff = jnp.where(r_ea_diff == 0, 1e-10, r_ea_diff)
        r_ea = jnp.linalg.norm(r_ea_diff, axis=-1)
        r_ea_mask = mol.active_electrons[:, None] * mol.active_nuclears[None, :]
        r_ee_diff = el_pos[:, None, :] - el_pos[None, :, :]
        r_ee_diff = jnp.where(r_ee_diff == 0, 1e-10, r_ee_diff)
        r_ee = jnp.linalg.norm(r_ee_diff, axis=-1)

        r_ee_mask = jnp.triu(
            mol.active_electrons[:, None] * mol.active_electrons[None, :], k=1
        )

        r_aa_diff = (
            mol.nuclear_positions[:, None, :] - mol.nuclear_positions[None, :, :]
        )
        r_aa_diff = jnp.where(r_aa_diff == 0, 1e-10, r_aa_diff)
        r_aa = jnp.linalg.norm(r_aa_diff, axis=-1)
        z_aa = mol.nuclear_charges[:, None] * mol.nuclear_charges[None, :]
        r_aa_mask = jnp.triu(
            mol.active_nuclears[:, None] * mol.active_nuclears[None, :], k=1
        )
        z_aa = mol.nuclear_charges[:, None] * mol.nuclear_charges[None, :]

        r_ee = jnp.where(r_ee == 0, 1e-10, r_ee)
        r_ea = jnp.where(r_ea == 0, 1e-10, r_ea)
        r_aa = jnp.where(r_aa == 0, 1e-10, r_aa)

        v_ee = jnp.sum((1 / r_ee) * r_ee_mask)

        v_ea = -jnp.sum((mol.nuclear_charges / r_ea) * r_ea_mask)
        v_aa = jnp.sum((z_aa / r_aa) * r_aa_mask)

        return v_ee + v_ea + v_aa
