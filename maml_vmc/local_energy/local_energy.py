from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp

import haiku as hk

from maml_vmc.local_energy.laplacian import get_kinetic_energy_v


class ClippingState(NamedTuple):
    center: jnp.ndarray
    width: jnp.ndarray


def clip_local_energy(E_loc, clipping_state: ClippingState):
    return (
        clipping_state.center
        + jnp.tanh((E_loc - clipping_state.center) / clipping_state.width)
        * clipping_state.width
    )


# @partial(jax.jit, static_argnames=["model", "potential_fn"])
def local_energy(inputs, model, model_params, potential_fn, potential_param):
    potential_op = potential_fn(inputs, potential_param)
    E_kin = get_kinetic_energy_v(inputs, model, model_params)
    return potential_op + E_kin


@partial(jax.custom_vjp, nondiff_argnums=(2, 4))
def local_energy_batch(
    psi_batch,
    input_batch,
    model,
    model_params,
    potential_fn,
    potential_param,
    clipping_state: ClippingState,
):

    local_energies = jax.vmap(local_energy, in_axes=(0, None, None, None, None))(
        input_batch,
        model,
        model_params,
        potential_fn,
        potential_param,
    )

    return local_energies


def local_energy_batch_fwd(
    psi_batch,
    input_batch,
    model,
    model_params,
    potential_fn,
    potential_params,
    clipping_state: ClippingState,
):

    le_batch = local_energy_batch(
        psi_batch,
        input_batch,
        model,
        model_params,
        potential_fn,
        potential_params,
        clipping_state,
    )
    return le_batch, (le_batch, psi_batch, clipping_state)


def local_energy_batch_bwd(model, potential_fn, ctx, ctangents):

    le_batch, psi_batch, clipping_state = ctx
    clipped_local_energies = clip_local_energy(le_batch, clipping_state)
    clipped_local_energies = jnp.where(
        jnp.isnan(clipped_local_energies),
        jnp.nanmean(clipped_local_energies) * jnp.ones_like(clipped_local_energies),
        clipped_local_energies,
    )
    clipped_mean = jnp.nanmean(clipped_local_energies)
    g = (clipped_local_energies - clipped_mean) * ctangents
    return (g, None, None, None, None)


local_energy_batch.defvjp(local_energy_batch_fwd, local_energy_batch_bwd)


def grad_loss_func(
    model,
    model_params_trainable,
    model_params_non_trainable,
    sampled_batch,
    potential_fn,
    system_state,
    clipping_state,
):
    """
    Original DeepErwin implementation of gradient
    https://github.com/mdsunivie/deeperwin
    """
    model_params = hk.data_structures.merge(
        model_params_trainable, model_params_non_trainable
    )

    @jax.custom_jvp
    def total_energy(model_params, aux_params):
        (input_batch,) = aux_params
        E_loc = jax.vmap(local_energy, in_axes=(0, None, None, None, None))(
            input_batch,
            model,
            model_params,
            potential_fn,
            system_state,
        )
        E_loc = jnp.where(
            jnp.isnan(E_loc), jnp.nanmean(E_loc) * jnp.ones_like(E_loc), E_loc
        )
        return jnp.mean(E_loc), E_loc

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        model_params, (input_batch,) = primals
        E_mean, E_loc = total_energy(*primals)
        log_psi_func_simplified = lambda p: jax.vmap(model.apply, in_axes=(None, 0))(
            p, input_batch
        )
        E_loc_c = clip_local_energy(E_loc, clipping_state)
        E_loc_c_mean = jnp.mean(E_loc_c)
        psi_primal, psi_tan = jax.jvp(
            log_psi_func_simplified, primals[:1], tangents[:1]
        )
        primals_out = (E_mean, E_loc)
        tangents_out = (
            jnp.dot(psi_tan, E_loc_c - E_loc_c_mean) / len(E_loc_c),
            E_loc_c,
        )
        return primals_out, tangents_out

    grads, E_batch = jax.grad(total_energy, argnums=0, has_aux=True)(
        model_params, (sampled_batch,)
    )
    return grads, E_batch
