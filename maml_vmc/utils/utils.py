import jax
import jax.numpy as jnp


def reinit_layer(layer, rng_key, out_shape=None, ones=False):
    """
    Reinitialize the weights of a layer
    :param layer: The layer to reinitialize
    :param rng_key: The random key to use for reinitialization
    :return: The reinitialized layer
    """
    if out_shape is None:
        out_shape = layer["w"].shape[-1]
    init = jax.nn.initializers.glorot_normal()
    w_shape = (layer["w"].shape[0], out_shape)
    w = init(rng_key, w_shape)
    if ones:
        w = jnp.zeros_like(w)
    # w = jnp.zeros((w.shape[0], w.shape[1]))
    b = jnp.zeros((out_shape,))
    return {"w": w, "b": b}
