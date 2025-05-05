from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.lax import scan, fori_loop


class MetropolisWalkerState(NamedTuple):
    """State of the walker"""

    position: jnp.ndarray
    spin: jnp.ndarray
    values: jnp.ndarray
    prng_key: jax.random.PRNGKey
    accepted: jnp.ndarray = None
    step_size: float = 0.01
    age: int = 0


class MetropolisWalker:

    @staticmethod
    def init_position(prng_key, num_pos_dims, el_ion_mapping, R):
        x = jax.random.normal(prng_key, (num_pos_dims))
        x = x.reshape(-1, 3)
        el_ion_pos = R[el_ion_mapping]
        x = x + el_ion_pos
        return x.reshape(-1)

    @staticmethod
    def init(
        prng_key: jax.random.PRNGKey,
        num_pos_dims: int,
        num_spin_dims: int,
        el_ion_mapping: jnp.ndarray,
        R: jnp.ndarray,
        eval_fn,
        init_step_size: float = 0.01,
    ) -> MetropolisWalkerState:
        prng_key, sub_key = jax.random.split(prng_key)
        pos = MetropolisWalker.init_position(sub_key, num_pos_dims, el_ion_mapping, R)
        prng_key, sub_key = jax.random.split(prng_key)
        spin = jax.random.choice(sub_key, jnp.array([-1, 1]), shape=(num_spin_dims,))
        values = eval_fn(pos, spin)
        return MetropolisWalkerState(
            position=pos,
            values=values,
            prng_key=prng_key,
            spin=spin,
            step_size=init_step_size,
            accepted=jnp.ones((), dtype=jnp.bool_),
        )

    @staticmethod
    def move(
        state: MetropolisWalkerState,
        eval_fn,
    ) -> MetropolisWalkerState:

        prng_key, sub_key1, sub_key2 = jax.random.split(state.prng_key, 3)

        old_values = state.values
        old_pos = state.position
        old_spin = state.spin
        # move the walker
        new_pos = (
            state.position
            + jax.random.normal(
                sub_key1, state.position.shape, dtype=state.position.dtype
            )
            * state.step_size
        )

        new_spin = jax.random.choice(
            sub_key1, jnp.array([-1, 1]), shape=(state.spin.shape)
        )

        new_values = eval_fn(new_pos, new_spin)

        prop_ratio = jnp.exp(new_values - old_values)
        accept = jax.random.uniform(sub_key2) < jnp.minimum(1.0, prop_ratio)
        is_too_old = state.age >= 10000000
        do_accept = jnp.logical_or(accept, is_too_old)
        new_pos = jnp.where(do_accept, new_pos, old_pos)
        new_spin = jnp.where(do_accept, new_spin, old_spin)
        new_values = jnp.where(do_accept, new_values, old_values)
        new_age = jnp.where(do_accept, 0, state.age + 1)
        ms = MetropolisWalkerState(
            position=new_pos,
            values=new_values,
            prng_key=prng_key,
            spin=new_spin,
            accepted=do_accept,
            step_size=state.step_size,
            age=new_age,
        )
        return (ms, ms)

    @staticmethod
    def update_values(state: MetropolisWalkerState, eval_fn):
        new_values = eval_fn(state.position, state.spin)
        return MetropolisWalkerState(
            position=state.position,
            values=new_values,
            prng_key=state.prng_key,
            spin=state.spin,
            accepted=state.accepted,
            step_size=state.step_size,
            age=state.age,
        )


# @partial(jax.jit, static_argnames=["eval_fn"])
def metropolis_move_batch(current_state: MetropolisWalkerState, eval_fn):
    current_state, _ = jax.vmap(
        MetropolisWalker.move,
        in_axes=(0, None),
    )(current_state, eval_fn)
    alpha = jnp.where(current_state.accepted.mean() < 0.5, 0.95, 1.05)
    new_stepsize = current_state.step_size * alpha
    current_state = MetropolisWalkerState(
        position=current_state.position,
        values=current_state.values,
        prng_key=current_state.prng_key,
        spin=current_state.spin,
        accepted=current_state.accepted,
        step_size=new_stepsize,
        age=current_state.age,
    )
    return current_state, current_state


def run_metropolis_batch(
    prng_key: jax.random.PRNGKey,
    num_walkers: int,
    num_pos_dims: int,
    num_spin_dims: int,
    burn_in: int,
    num_steps: int,
    step_size: float,
    el_ion_mapping,
    R,
    eval_fn,
    init_walker_state: MetropolisWalkerState = None,
):
    if init_walker_state is None:
        prng_keys = jax.random.split(prng_key, num_walkers)
        init_walker_state = jax.vmap(
            MetropolisWalker.init,
            in_axes=(0, None, None, None, None, None, None),
        )(
            prng_keys,
            num_pos_dims,
            num_spin_dims,
            el_ion_mapping,
            R,
            eval_fn,
            step_size,
        )
    # update
    init_walker_state = jax.vmap(
        MetropolisWalker.update_values,
        in_axes=(0, None),
    )(init_walker_state, eval_fn)
    walker_state = fori_loop(
        0,
        burn_in,
        lambda i, current_state: metropolis_move_batch(current_state, eval_fn)[0],
        init_walker_state,
    )

    last_state, y_state = scan(
        lambda current_state, _: metropolis_move_batch(current_state, eval_fn),
        init=walker_state,
        xs=None,
        length=num_steps,
    )
    return y_state, last_state


def run_metropolis(
    prng_key: jax.random.PRNGKey,
    num_pos_dims: int,
    num_spin_dims: int,
    burn_in: int,
    num_steps: int,
    step_size: float,
    eval_fn,
):
    prng_key, sub_key = jax.random.split(prng_key)
    init_walker_state = MetropolisWalker.init(
        sub_key, num_pos_dims, num_spin_dims, eval_fn
    )

    # walker_state: MetropolisWalkerState = fori_loop(
    #     0,
    #     burn_in,
    #     lambda i, state: MetropolisWalker.move(state, step_size, eval_fn)[0],
    #     init_walker_state,
    # )

    walker_state = fori_loop(
        0,
        burn_in,
        lambda i, current_state: MetropolisWalker.move(
            current_state, step_size, eval_fn
        ),
        init_val=init_walker_state,
    )

    # TODO: add uncorrelated sampling (add additional steps between samples)
    last_state, y_state = scan(
        lambda current_state, _: MetropolisWalker.move(
            current_state, step_size, eval_fn
        ),
        init=walker_state,
        xs=None,
        length=num_steps,
    )

    return y_state, last_state
