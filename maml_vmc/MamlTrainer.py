# or locally with
from functools import partial
from typing import Callable, Literal, NamedTuple, Tuple
import jax

import jax.numpy as jnp
import loguru
import optax
from tqdm import tqdm
from maml_vmc import lijax
import haiku as hk


from maml_vmc.lijax.utils import load_training_state
from maml_vmc.maml_optimization import (
    ClippingState,
    eval_system,
    inner_loop,
)
from maml_vmc.sampler.MetropolisSampler import MetropolisWalkerState
from maml_vmc.sampler.utils import SystemState
from maml_vmc.utils.haiku_utils import get_hk_regex_keys
from maml_vmc.utils.utils import reinit_layer


class TrainingState(NamedTuple):
    model_params: hk.Params
    optimizer_state: optax.OptState
    inner_optimizer_state: optax.OptState
    prng_key: jax.random.PRNGKey
    step: int
    clipping_state: ClippingState
    walker_state: MetropolisWalkerState


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")


class MamlTrainer(lijax.Trainer):
    def __init__(
        self,
        num_steps: int,
        num_inner_steps: int,
        num_inner_steps_validation: int,
        num_systems: int,
        num_systems_validation: int,
        model: lijax.Module,
        optimizer: lijax.Optimizer,
        inner_optimizer: lijax.Optimizer,
        data_sampler: lijax.DataSampler,
        val_data_sampler: lijax.DataSampler,
        seed: int,
        eval_every: int = 1,
        store_every: int = 50,
        inner_loop_freezing_keys_re: list = [],
        ckpt_path: str = None,
        reinit_layers: bool = False,
        optimization_strategy: Literal["reptile", "direct", "fomaml"] = "reptile",
        trans_weights: str = None,
        num_batches: int = 1,
        reinit_ones: bool = False,
        storing_epochs: list = [],
        trans_num_electrons: int = 0,
        trans_num_dets: int = 0,
    ):
        super().__init__(num_steps=num_steps)

        self.prng_key = jax.random.PRNGKey(seed)
        self.model_p = model
        self.model = hk.without_apply_rng(
            hk.transform(lambda x: self.model_p.get_model()(x))
        )
        self.raw_optimizer = optimizer
        self.optimizer = optimizer.get_optimizer()
        self.innner_optimizer = inner_optimizer.get_optimizer()
        self.data_sampler = data_sampler
        self.num_inner_steps = num_inner_steps
        self.num_inner_steps_validation = num_inner_steps_validation
        self.num_systems = num_systems
        self.num_systems_validation = num_systems_validation
        self.trans_weights = trans_weights
        self.eval_every = eval_every
        self.inner_loop_freezing_keys_re = inner_loop_freezing_keys_re
        self.ckpt_path = ckpt_path
        self.store_every = store_every
        self.val_data_sampler = val_data_sampler
        self.optimization_strategy = optimization_strategy
        self.reinit_layers = reinit_layers
        self.num_batches = num_batches
        self.reinit_ones = reinit_ones
        self.storing_epochs = storing_epochs
        self.trans_num_electrons = trans_num_electrons
        self.trans_num_dets = trans_num_dets

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "model",
            "meta_optimizer",
            "inner_optimizer",
            "data_sampler",
            "potential_fn",
            "num_inner_steps",
            "num_systems",
            "optimization_strategy",
            "num_batches",
        ],
    )
    def step(
        model: hk.Transformed,
        model_params_trainable,
        model_params_non_trainable,
        meta_optimizer: optax.GradientTransformation,
        inner_optimizer: optax.GradientTransformation,
        data_sampler: lijax.DataSampler,
        potential_fn: Callable,
        prng_key: jax.random.PRNGKey,
        meta_optimizer_state,
        inner_optimizer_state,
        num_inner_steps: int,
        num_systems: int,
        clipping_state,
        walker_state: MetropolisWalkerState = None,
        optimization_strategy: str = "reptile",
        num_batches: int = 1,
    ) -> Tuple[dict, dict, jnp.ndarray, jnp.ndarray, NamedTuple]:
        # sample systems
        system_prng_keys = jax.random.split(prng_key, num_systems)
        # idx = jnp.arange(num_systems)
        system_states = jax.vmap(data_sampler.sample_system, in_axes=(0, None))(
            system_prng_keys, -1
        )
        system_states: SystemState = system_states
        fi_inner_loop_energies, final_loop_state = jax.vmap(
            inner_loop,
            in_axes=(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                None,
                None,
                None,
                None,
                None,
            ),
        )(
            model,
            model_params_trainable,
            model_params_non_trainable,
            inner_optimizer,
            data_sampler,
            potential_fn,
            prng_key,
            system_states,
            inner_optimizer_state,
            num_inner_steps,
            clipping_state,
            walker_state,
            num_batches,
        )
        # inner loop and fomaml step
        if optimization_strategy in ["reptile", "fomaml"]:
            if optimization_strategy == "fomaml":
                meta_grads = jax.tree.map(
                    lambda x: jnp.mean(x, axis=0), final_loop_state.last_grads
                )
            elif optimization_strategy == "reptile":
                grads = jax.tree.map(
                    lambda x, y: x - y,
                    model_params_trainable,
                    final_loop_state.inner_model_params_trainable,
                )
                meta_grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), tree=grads)
            else:
                raise ValueError("Unknown optimization")

            model_params = hk.data_structures.merge(
                model_params_trainable, model_params_non_trainable
            )
            meta_updates, new_meta_optimizer_state = meta_optimizer.update(
                meta_grads, meta_optimizer_state, model_params
            )
            new_model_params = optax.apply_updates(model_params, meta_updates)
            new_inner_optimization_state = inner_optimizer_state
            new_clipping_state = clipping_state
            new_walker_state = walker_state

        elif optimization_strategy == "direct":
            # only one system
            final_loop_state_s = jax.tree.map(lambda x: x[0], final_loop_state)
            new_model_params = hk.data_structures.merge(
                final_loop_state_s.inner_model_params_trainable,
                model_params_non_trainable,
            )
            new_meta_optimizer_state = meta_optimizer_state
            new_inner_optimization_state = final_loop_state_s.inner_optimizer_state
            new_clipping_state = final_loop_state_s.clipping_state
            new_walker_state = final_loop_state_s.walker_state
        else:
            raise ValueError("Unknown optimization")

        return (
            new_model_params,
            new_meta_optimizer_state,
            new_inner_optimization_state,
            new_clipping_state,
            new_walker_state,
            fi_inner_loop_energies,
            final_loop_state.current_energy,
            final_loop_state.current_energy_std,
            system_states,
        )

    def fit(self):
        self.prng_key, prng_key = jax.random.split(self.prng_key)

        prng_key, sub_key = jax.random.split(prng_key)
        # initialize model and optimizer
        model_params = self.model.init(sub_key, self.data_sampler.get_fake_datapoint())

        # model, model_params = get_app()
        # self.model = model

        if self.trans_weights is not None:
            t_s: TrainingState = load_training_state(self.trans_weights)
            if self.optimization_strategy == "direct":
                model_params = t_s.model_params
            if self.optimization_strategy == "direct" and self.reinit_layers:
                model_params["DeepErwin/~/BackflowFactor/~/bf_factor_up/linear_3"] = (
                    reinit_layer(
                        model_params[
                            "DeepErwin/~/BackflowFactor/~/bf_factor_down/linear_3"
                        ],
                        sub_key,
                        self.trans_num_dets * self.trans_num_electrons,
                        ones=self.reinit_ones,
                    )
                )
                prng_key, sub_key = jax.random.split(prng_key)
                model_params["DeepErwin/~/BackflowFactor/~/bf_factor_down/linear_3"] = (
                    reinit_layer(
                        model_params[
                            "DeepErwin/~/BackflowFactor/~/bf_factor_down/linear_3"
                        ],
                        sub_key,
                        self.trans_num_dets * self.trans_num_electrons,
                        ones=self.reinit_ones,
                    )
                )
                prng_key, sub_key = jax.random.split(prng_key)

        # get num_params
        flat_params = jax.tree.flatten(model_params)[0]
        num_params = jax.tree.map(lambda x: x.size, flat_params)
        num_params = jnp.sum(jnp.array(num_params))
        print(f"number of parameters: {num_params}")

        # get freezing keys for inner loop by regex
        self.inner_loop_freezing_keys = get_hk_regex_keys(
            model_params, self.inner_loop_freezing_keys_re
        )

        # partition trainable and non trainable params
        inner_trainable_params, inner_non_trainable_params = (
            hk.data_structures.partition(
                lambda m, n, p: m + "." + n not in self.inner_loop_freezing_keys,
                model_params,
            )
        )

        # initialize optimizer for meta and inner loop optimization
        # add and remove learning rates from model params for optimizer initialization
        optimizer_state = self.optimizer.init(model_params)
        inner_optimizer_state = self.innner_optimizer.init(inner_trainable_params)

        # initialize training state and load checkpoint if available
        training_state = TrainingState(
            model_params=model_params,
            optimizer_state=optimizer_state,
            inner_optimizer_state=inner_optimizer_state,
            prng_key=prng_key,
            step=0,
            clipping_state=ClippingState(
                center=jnp.zeros(()),
                width=1000 * jnp.ones(()),
            ),
            walker_state=None,
        )

        if self.ckpt_path is not None:
            training_state = load_training_state(self.ckpt_path)
            self.logger.set_current_step(training_state.step)

        for step in range(training_state.step, self.num_steps):

            prng_key, sub_key = jax.random.split(training_state.prng_key)

            # partition trainable and non trainable params
            inner_trainable_params, inner_non_trainable_params = (
                hk.data_structures.partition(
                    lambda m, n, p: m + "." + n not in self.inner_loop_freezing_keys,
                    training_state.model_params,
                )
            )

            (
                new_model_params,
                new_meta_optimizer_state,
                new_inner_optimization_state,
                new_clipping_state,
                new_walker_state,
                inner_loop_energies,
                energies,
                energies_std,
                system_states,
            ) = MamlTrainer.step(
                model=self.model,
                model_params_trainable=training_state.model_params,
                model_params_non_trainable=inner_non_trainable_params,
                meta_optimizer=self.optimizer,
                inner_optimizer=self.innner_optimizer,
                data_sampler=self.data_sampler,
                potential_fn=self.data_sampler.potential_fn,
                prng_key=sub_key,
                meta_optimizer_state=training_state.optimizer_state,
                inner_optimizer_state=training_state.inner_optimizer_state,
                num_inner_steps=self.num_inner_steps,
                num_systems=self.num_systems,
                clipping_state=training_state.clipping_state,
                walker_state=training_state.walker_state,
                optimization_strategy=self.optimization_strategy,
                num_batches=self.num_batches,
            )

            errors = energies - system_states.ref_energy
            print_max = 6

            self.logger.log_metrics(
                {
                    "all_std": (energies_std.mean(), True),
                    "mean_error": (errors.mean(), True),
                    "clipped_E": (new_clipping_state.center, True),
                }
            )

            # update training state

            if self.reinit_layers:
                prng_key, sub_key = jax.random.split(prng_key)
                new_model_params[
                    "DeepErwin/~/BackflowFactor/~/bf_factor_up/linear_3"
                ] = reinit_layer(
                    new_model_params[
                        "DeepErwin/~/BackflowFactor/~/bf_factor_up/linear_3"
                    ],
                    sub_key,
                    ones=self.reinit_ones,
                )
                prng_key, sub_key = jax.random.split(prng_key)
                new_model_params[
                    "DeepErwin/~/BackflowFactor/~/bf_factor_down/linear_3"
                ] = reinit_layer(
                    new_model_params[
                        "DeepErwin/~/BackflowFactor/~/bf_factor_down/linear_3"
                    ],
                    sub_key,
                    ones=self.reinit_ones,
                )
            training_state = TrainingState(
                model_params=new_model_params,
                optimizer_state=new_meta_optimizer_state,
                inner_optimizer_state=new_inner_optimization_state,
                prng_key=prng_key,
                step=step + 1,
                clipping_state=new_clipping_state,
                walker_state=new_walker_state,
            )

            if (step % self.store_every == 0 and step != 0) or (
                step + 1 in self.storing_epochs
            ):
                self.logger.store_training_state(training_state)
            self.logger.step_finished()

    def validate(self):
        assert self.ckpt_path is not None, "ckpt_path must be set for validation"
        loguru.logger.info("loading training state: " + self.ckpt_path)
        prng_key, sub_key = jax.random.split(self.prng_key)
        training_state: TrainingState = load_training_state(self.ckpt_path)
        system_state: SystemState = self.val_data_sampler.sample_system(sub_key, 0)
        model_params = training_state.model_params
        prng_key, sub_key = jax.random.split(prng_key)
        _, walker_state = self.val_data_sampler.sample_data(
            prng_key,
            system_state,
            self.model,
            model_params,
            init_walker_state=None,
        )

        local_energies = []
        current_pos = []

        bar = tqdm(range(1000))
        for i in bar:

            prng_key, sub_key = jax.random.split(prng_key)

            les, walker_state = eval_system(
                self.model,
                model_params,
                self.val_data_sampler,
                self.val_data_sampler.potential_fn,
                sub_key,
                system_state,
                walker_state,
            )

            local_energies.append(les)
            current_pos.append(walker_state.position)
        current_mean = jnp.nanmean(jnp.concatenate(local_energies, axis=0))
        current_std = jnp.nanstd(jnp.concatenate(local_energies, axis=0))
        error = current_mean - system_state.ref_energy
        print(
            f"mean: {current_mean}Ha, std: {current_std}, ref: {system_state.ref_energy}Ha"
        )
        print("=====================================")
        print(f"error: {error*1000:.4f}mHa")
