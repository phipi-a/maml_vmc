import jax
import jax.numpy as jnp


def print_jax(text, *args):
    if not isinstance(text, str):
        args = [text]
        text = ""
    list_args = list(args)
    text = str(text) + " " + "".join(["{} " for _ in list_args])

    jax.debug.print(text, *list_args, ordered=True)


def print_contain_nan(name, x):
    # check if any element is a tree
    if jax.tree_util.tree_flatten(x)[0]:
        t = jax.tree.map(lambda x: jnp.any(jnp.isnan(x)), x)
        # is any element in the tree True?
        any_nan = jax.tree.reduce(lambda x, y: jnp.logical_or(x, y), t)
        print_jax(name, any_nan)
    else:
        print_jax(name, jnp.any(jnp.isnan(x)))


def get_total_mean(x):
    x = jax.tree.map(lambda x: jnp.nanmean(x, dtype=jnp.float32), x)
    x = jax.tree.reduce(lambda x, y: jnp.nanmean(jnp.stack([x, y], axis=0)), x)
    return x
