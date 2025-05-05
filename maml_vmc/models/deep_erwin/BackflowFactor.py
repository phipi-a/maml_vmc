import jax
from maml_vmc import lijax
from maml_vmc.models.utils.FCNN import FCNN
from maml_vmc.models.deep_erwin.utils import DOWN, UP
import jax.numpy as jnp
import haiku as hk

from maml_vmc.sampler.MoleculeDataSampler import (
    MoleculeFeatures,
)


class BackflowFactor(lijax.Module):
    def __init__(
        self,
        n_max_electrons: int,
        n_dets: int = 32,
        name: str = "BackflowFactor",
        n_hidden_bf_factor: list[int] = [40, 20, 20],
    ):
        super().__init__(name=name)
        self.n_dets = n_dets
        bf_factor_up_shape = n_hidden_bf_factor + [n_dets * n_max_electrons]
        bf_factor_down_shape = n_hidden_bf_factor + [n_dets * n_max_electrons]
        self.n_max_electrons = n_max_electrons
        self.bf_factor_up_net_p = FCNN(
            bf_factor_up_shape,
            activation_fn=jax.nn.tanh,
            use_second_last_layer_activation=False,
            name="bf_factor_up",
        )
        self.bf_factor_down_net_p = FCNN(
            bf_factor_down_shape,
            activation_fn=jax.nn.tanh,
            use_second_last_layer_activation=False,
            name="bf_factor_down",
        )

    def init_modules(self):
        self.bf_factor_up_net = self.bf_factor_up_net_p.get_model()
        self.bf_factor_down_net = self.bf_factor_down_net_p.get_model()

    def __call__(
        self,
        embeddings,
        mol: MoleculeFeatures,
    ) -> jax.Array:
        # basic feature
        n_el = embeddings.shape[0]
        up_mask = (mol.electron_spins == UP)[:, None, None]
        down_mask = (mol.electron_spins == DOWN)[:, None, None]
        bf_factor_up_p = self.bf_factor_up_net(embeddings).reshape(
            n_el, self.n_dets, n_el
        )
        # randomize order of electrons
        key_up, key_down = jax.random.split(mol.random_key)
        # perm_up = jax.random.permutation(key_up, n_el)
        # perm_down = jax.random.permutation(key_down, n_el)
        bf_factor_up = bf_factor_up_p[:, :, :] * up_mask
        bf_factor_down_p = self.bf_factor_down_net(embeddings).reshape(
            n_el, self.n_dets, n_el
        )
        bf_factor_down = bf_factor_down_p[:, :, :] * down_mask
        bf_factor_down = jnp.roll(bf_factor_down, shift=-mol.n_up, axis=0)

        scale = hk.get_parameter(
            "scale",
            shape=(),
            init=hk.initializers.Constant(-2.0),
        )
        bf_factor_up = 1 + (jnp.exp(scale) * bf_factor_up)
        bf_factor_down = 1 + (jnp.exp(scale) * bf_factor_down)

        # bf_factor_up: [n_up x n_dets*n_up_orb]
        # bf_factor_down: [n_down x n_dets*n_down_orb]
        # bf_factor_down = bf_factor_down.reshape(n_el, self.n_dets, n_el)
        # bf_factor_up: [n_up x n_dets x n_up_orb]
        # bf_factor_down: [n_down x n_dets x n_down_orb]
        bf_factor_up = jnp.transpose(bf_factor_up, (1, 0, 2))
        bf_factor_down = jnp.transpose(bf_factor_down, (1, 0, 2))
        # bf_factor_up: [n_dets x n_up x n_up_orb]
        # bf_factor_down: [n_dets x n_down x n_down_orb]
        return bf_factor_up, bf_factor_down
