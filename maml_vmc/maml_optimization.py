from functools import partial
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
import optax
from maml_vmc import lijax
import haiku as hk

# TODO: local_energy2 or local_energy?
from maml_vmc.local_energy.local_energy import (
    ClippingState,
    clip_local_energy,
    local_energy_batch,
)
from maml_vmc.sampler.MetropolisSampler import MetropolisWalkerState


class InnerLoopState(NamedTuple):
    inner_model_params_trainable: hk.Params
    inner_model_params_non_trainable: hk.Params
    inner_optimizer_state: optax.OptState
    prng_key: jax.random.PRNGKey
    current_energy: jnp.ndarray
    current_energy_std: jnp.ndarray
    clipping_state: ClippingState
    last_grads: jnp.ndarray
    walker_state: MetropolisWalkerState = None


@partial(jax.jit, static_argnames=["model", "potential_fn"])
def eval_local_energy(
    model: hk.Transformed,
    model_params: dict,
    batch,
    potential_fn: Callable,
    potential_param,
    clipping_state: ClippingState,
):
    psi_out_batch = jax.vmap(model.apply, in_axes=(None, 0))(model_params, batch)

    local_energies = local_energy_batch(
        psi_out_batch,
        batch,
        model,
        model_params,
        potential_fn,
        potential_param,
        clipping_state,
    )
    return local_energies


def get_clipping_state(E_loc):
    center = jnp.nanmean(E_loc)
    width = jnp.nanstd(E_loc) * 5
    return ClippingState(center, width)


def loss_fn(
    model: hk.Transformed,
    model_params_trainable: dict,
    model_paramas_non_trainable: dict,
    batch,
    potential_fn: Callable,
    potential_param,
    local_energy_clipping_state: ClippingState,
):
    model_params = hk.data_structures.merge(
        model_params_trainable, model_paramas_non_trainable
    )
    local_energies = eval_local_energy(
        model,
        model_params,
        batch,
        potential_fn,
        potential_param,
        local_energy_clipping_state,
    )
    return jnp.nanmean(local_energies), local_energies


def evaluate(
    model: hk.Transformed,
    model_params: dict,
    data_sampler: lijax.DataSampler,
    potential_fn: Callable,
    system_state,
    prng_key: jax.random.PRNGKey,
    local_energy_clipping_state: ClippingState,
    init_walker_state: MetropolisWalkerState = None,
):
    sampled_batch, walker_state = data_sampler.sample_data(
        prng_key, system_state, model, model_params, init_walker_state
    )

    le_batch = eval_local_energy(
        model,
        model_params,
        sampled_batch,
        potential_fn,
        system_state,
        local_energy_clipping_state,
    )

    return (
        le_batch.mean(),
        le_batch,
        walker_state,
    )


def inner_step(
    model: hk.Transformed,
    inner_optimizer: optax.GradientTransformation,
    data_sampler: lijax.DataSampler,
    potential_fn: Callable,
    system_state,
    num_batches: int,
    inner_loop_state: InnerLoopState,
):

    sub_key, prng_key = jax.random.split(inner_loop_state.prng_key)
    model_params = hk.data_structures.merge(
        inner_loop_state.inner_model_params_trainable,
        inner_loop_state.inner_model_params_non_trainable,
    )

    class BatchCarry(NamedTuple):
        inner_model_params_trainable: hk.Params
        optimzer_state: optax.OptState

    def update_step(batch_carry: BatchCarry, sampled_batch):

        (energy, local_energies), gradients = jax.value_and_grad(
            loss_fn, argnums=1, has_aux=True
        )(
            model,
            batch_carry.inner_model_params_trainable,
            inner_loop_state.inner_model_params_non_trainable,
            sampled_batch,
            potential_fn,
            system_state,
            inner_loop_state.clipping_state,
        )

        # gradients, local_energies = grad_loss_func(
        #     model,
        #     batch_carry.inner_model_params_trainable,
        #     inner_loop_state.inner_model_params_non_trainable,
        #     sampled_batch,
        #     potential_fn,
        #     system_state,
        #     inner_loop_state.clipping_state,
        # )
        gradients = jax.tree.map(
            lambda x: jnp.where(jnp.isnan(x), jnp.zeros_like(x), x), gradients
        )
        updates, new_inner_optimizer_state = inner_optimizer.update(
            gradients,
            batch_carry.optimzer_state,
            batch_carry.inner_model_params_trainable,
        )
        new_inner_model_params = optax.apply_updates(
            batch_carry.inner_model_params_trainable, updates
        )

        return (
            BatchCarry(
                inner_model_params_trainable=new_inner_model_params,
                optimzer_state=new_inner_optimizer_state,
            ),
            (local_energies, gradients),
        )

    sampled_batches, new_walker_state = data_sampler.sample_data(
        sub_key,
        system_state,
        model,
        model_params,
        inner_loop_state.walker_state,
    )
    sampled_batches = jax.tree.map(
        lambda x: x.reshape(num_batches, -1, *x.shape[1:]), sampled_batches
    )

    final_batch_carry, (local_energies_batch, gradients_batch) = jax.lax.scan(
        lambda carry, x: update_step(carry, x),
        BatchCarry(
            inner_model_params_trainable=inner_loop_state.inner_model_params_trainable,
            optimzer_state=inner_loop_state.inner_optimizer_state,
        ),
        xs=sampled_batches,
    )
    local_energies = local_energies_batch.flatten()
    energy = jnp.nanmean(local_energies)
    gradients = jax.tree.map(lambda x: x[0], gradients_batch)
    # update clipping state
    new_clipping_state = get_clipping_state(
        clip_local_energy(local_energies, inner_loop_state.clipping_state)
    )

    new_inner_model_params, new_inner_optimizer_state = (
        final_batch_carry.inner_model_params_trainable,
        final_batch_carry.optimzer_state,
    )
    return (
        InnerLoopState(
            new_inner_model_params,
            inner_loop_state.inner_model_params_non_trainable,
            new_inner_optimizer_state,
            prng_key,
            energy,
            jnp.nanstd(local_energies),
            new_clipping_state,
            gradients,
            new_walker_state,
        ),
        energy,
    )


def inner_loop(
    model: hk.Transformed,
    model_params_trainable,
    model_params_non_trainable,
    inner_optimizer: optax.GradientTransformation,
    data_sampler: lijax.DataSampler,
    potential_fn: Callable,
    prng_key: jax.random.PRNGKey,
    system_state,
    inner_optimizer_state,
    num_inner_steps: int,
    clipping_state: ClippingState,
    walker_state: MetropolisWalkerState = None,
    num_batches: int = 1,
):
    if walker_state is None:
        # if walker is not provided  get the initial walker state
        prng_key, sub_key = jax.random.split(prng_key)
        model_params = hk.data_structures.merge(
            model_params_trainable,
            model_params_non_trainable,
        )
        batch, walker_state = data_sampler.sample_data(
            sub_key, system_state, model, model_params
        )

    if num_inner_steps > 1:
        # inner loop
        def f(state, _):
            return inner_step(
                model,
                inner_optimizer,
                data_sampler,
                potential_fn,
                system_state,
                num_batches,
                state,
            )

        final_loop_state, inner_loop_energies = jax.lax.scan(
            f,
            InnerLoopState(
                model_params_trainable,
                model_params_non_trainable,
                inner_optimizer_state,
                prng_key,
                jnp.array(0.0),
                jnp.array(0.0),
                clipping_state,
                jax.tree.map(lambda x: jnp.zeros_like(x), model_params_trainable),
                walker_state,
            ),
            None,
            length=num_inner_steps,
        )
        total_final_params = hk.data_structures.merge(
            final_loop_state.inner_model_params_trainable, model_params_non_trainable
        )
    elif num_inner_steps == 1:
        # single inner loop step
        final_loop_state, energy = inner_step(
            model,
            inner_optimizer,
            data_sampler,
            potential_fn,
            system_state,
            num_batches,
            InnerLoopState(
                model_params_trainable,
                model_params_non_trainable,
                inner_optimizer_state,
                prng_key,
                jnp.array(0.0),
                jnp.array(0.0),
                clipping_state,
                jax.tree.map(lambda x: jnp.zeros_like(x), model_params_trainable),
                walker_state,
            ),
        )
        total_final_params = hk.data_structures.merge(
            final_loop_state.inner_model_params_trainable, model_params_non_trainable
        )
        inner_loop_energies = jnp.array([energy])
    else:
        # no inner loop steps
        total_final_params = hk.data_structures.merge(
            model_params_trainable, model_params_non_trainable
        )
        inner_loop_energies = jnp.array([])
        final_loop_state = InnerLoopState(
            model_params_trainable,
            model_params_non_trainable,
            inner_optimizer_state,
            prng_key,
            jnp.array(0.0),
            jnp.array(0.0),
            clipping_state,
            jax.tree.map(lambda x: jnp.zeros_like(x), model_params_trainable),
            walker_state,
        )

    return inner_loop_energies, final_loop_state


@partial(jax.jit, static_argnames=["model", "data_sampler", "potential_fn"])
def eval_system(
    model,
    model_params,
    data_sampler,
    potential_fn,
    prng_key,
    system_state,
    walker_state,
):
    energy, local_energies, walker_state = evaluate(
        model,
        model_params,
        data_sampler,
        potential_fn,
        system_state,
        prng_key,
        ClippingState(0.0, 1000.0),
        walker_state,
    )
    return local_energies, walker_state
