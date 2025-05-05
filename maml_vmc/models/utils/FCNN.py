from typing import Callable
import jax.numpy as jnp
import haiku as hk
import jax
from maml_vmc import lijax


class FCNN(lijax.Module):
    def __init__(
        self,
        output_dims: list[int],
        activation_fn: Callable = jax.nn.sigmoid,
        boundary_fn: Callable = None,
        norm_as_input: bool = False,
        name: str = None,
        use_last_layer_activation: bool = False,
        use_second_last_layer_activation: bool = True,
    ):
        super().__init__(name)
        self.output_dims = output_dims
        self.activation_fn = activation_fn
        self.boundary_fn = boundary_fn
        self.norm_as_input = norm_as_input
        self.use_last_layer_activation = use_last_layer_activation
        self.use_second_last_layer_activation = use_second_last_layer_activation

    def __call__(self, x):
        w_init = 0.1
        b_init = 0.0
        x_in = x
        glorot_initializer = hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )

        if self.norm_as_input:
            r = jnp.linalg.norm(x_in, axis=-1, keepdims=True)
            x = jnp.concatenate([x, r], axis=-1)
        for i, od in enumerate(self.output_dims[:-1]):
            x = hk.Linear(
                od,
                # w_init=hk.initializers.Constant(w_init),
                # b_init=hk.initializers.Constant(b_init),
                # w_init=glorot_initializer,
            )(x)
            if (
                not self.use_second_last_layer_activation
                and i == len(self.output_dims) - 2
            ):
                continue
            x = self.activation_fn(x)

        x = hk.Linear(
            self.output_dims[-1],
            # w_init=glorot_initializer,
            # w_init=hk.initializers.Constant(w_init),
            # b_init=hk.initializers.Constant(b_init),
        )(x)

        if self.activation_fn is not None and self.use_last_layer_activation:
            x = self.activation_fn(x)
        if self.boundary_fn is None:
            return x
        bc = self.boundary_fn(x_in)
        return bc * x
