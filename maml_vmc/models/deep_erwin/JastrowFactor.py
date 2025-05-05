import jax
from maml_vmc import lijax
from maml_vmc.models.utils.FCNN import FCNN
from maml_vmc.models.deep_erwin.utils import DOWN, UP, MMSMoleculeFeaturesBaseline
import haiku as hk
import jax.numpy as jnp


class JastrowFactor(lijax.Module):
    def __init__(
        self,
        name: str = "JastrowFactor",
        n_hidden_jastrow: list[int] = [40, 40],
    ):
        super().__init__(name=name)
        jastrow_shape = n_hidden_jastrow + [1]
        self.jastrow_factor_net_up_p = FCNN(
            jastrow_shape, activation_fn=jax.nn.tanh, name="jastrow_up"
        )
        self.jastrow_factor_net_down_p = FCNN(
            jastrow_shape, activation_fn=jax.nn.tanh, name="jastrow_down"
        )

    def init_modules(self):
        self.jastrow_factor_net_up = self.jastrow_factor_net_up_p.get_model()
        self.jastrow_factor_net_down = self.jastrow_factor_net_down_p.get_model()

    def __call__(
        self,
        embeddings,
        mol: MMSMoleculeFeaturesBaseline,
    ) -> jax.Array:
        mask_up = (mol.electron_spins == UP)[:, None]
        mask_down = (mol.electron_spins == DOWN)[:, None]
        jastrow_up = self.jastrow_factor_net_up(embeddings) * mask_up
        jastrow_down = self.jastrow_factor_net_down(embeddings) * mask_down
        n_up = mask_up.sum()
        n_down = mask_down.sum()

        scale = hk.get_parameter(
            "scale",
            shape=(),
            init=hk.initializers.Constant(0.0),
        )
        jastrow = (jastrow_up.sum() + jastrow_down.sum()) * jnp.exp(scale)
        return jastrow
