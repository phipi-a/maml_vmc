from dataclasses import asdict
from typing import Tuple
import jax
import jax.numpy as jnp


# @partial(jax.jit, static_argnames=["psi_model"])
def laplacian(batch, psi_model, psi_model_params):

    def f(psi_model_params, x, positions):
        # x.electron_positions = positions
        # copy named tuple to dict
        x = asdict(x)
        x["electron_positions"] = positions
        x = type(batch)(**x)
        return psi_model.apply(psi_model_params, x).sum()

    df_dx = jax.grad(f, argnums=2)

    def df_dxi(positions, x, mask, psi_model_params):
        o = jnp.dot(df_dx(psi_model_params, x, positions), mask)
        return o

    def d2f_dx2i(
        position, x, mask, psi_model_params
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        g1, g2 = jax.value_and_grad(df_dxi)(position, x, mask, psi_model_params)
        return g1, jnp.dot(g2, mask)

    mask = jnp.diag(jnp.ones(batch.electron_positions.shape[-1]))
    d2f_dx2 = jax.vmap(
        d2f_dx2i,
        in_axes=(None, None, 0, None),
    )
    derivative, laplacian = d2f_dx2(
        batch.electron_positions, batch, mask, psi_model_params
    )
    return derivative, laplacian


def get_kinetic_energy_own(batch, psi_model, psi_model_params):
    dv, lp = laplacian(batch, psi_model, psi_model_params)
    return -((1 / 4) * lp.sum()) - ((1 / 8) * (dv**2).sum())


def get_kinetic_energy(batch, psi_model, psi_model_params):
    n_coords = batch.electron_positions.shape[-1]
    eye = jnp.eye(n_coords, dtype=batch.electron_positions.dtype)

    def f(positions):
        # x.electron_positions = positions
        # copy named tuple to dict
        x = asdict(batch)
        x["electron_positions"] = positions
        x = type(batch)(**x)
        return psi_model.apply(psi_model_params, x).sum()

    grad_psi_func = lambda r: jax.grad(f, argnums=0)(r).flatten()

    def _loop_body(i, laplacian):
        g_i, G_ii = jax.jvp(
            grad_psi_func, (batch.electron_positions.flatten(),), (eye[i],)
        )
        return laplacian + G_ii[i] + 0.5 * g_i[i] ** 2

    return -0.25 * jax.lax.fori_loop(0, n_coords, _loop_body, 0.0)


def get_kinetic_energy_v(batch, psi_model, psi_model_params):
    n_coords = batch.electron_positions.shape[-1]
    eye = jnp.eye(n_coords, dtype=batch.electron_positions.dtype)

    def f(positions):
        # x.electron_positions = positions
        # copy named tuple to dict
        x = asdict(batch)
        x["electron_positions"] = positions
        x = type(batch)(**x)
        return psi_model.apply(psi_model_params, x).sum()

    grad_psi_func = lambda r: jax.grad(f, argnums=0)(r).flatten()

    def _loop_body(i):
        g_i, G_ii = jax.jvp(
            grad_psi_func, (batch.electron_positions.flatten(),), (eye[i],)
        )
        return G_ii[i] + 0.5 * g_i[i] ** 2

    o = jax.vmap(_loop_body, in_axes=0)(jnp.arange(n_coords))

    return -0.25 * o.sum()
