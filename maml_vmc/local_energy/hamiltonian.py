from maml_vmc.local_energy.laplacian import laplacian


# @partial(jax.jit, static_argnames=["psi_model", "potential_fn"])
def h_psi(psi_out, inputs, psi_model, psi_model_params, potential_fn, potential_param):
    lp = laplacian(inputs, psi_model, psi_model_params)

    potential_energy = potential_fn(inputs, potential_param) * psi_out
    kinetic_energy = -0.5 * lp
    return potential_energy + kinetic_energy
