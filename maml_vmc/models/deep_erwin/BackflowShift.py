import jax
from maml_vmc import lijax
from maml_vmc.models.utils.FCNN import FCNN
from maml_vmc.sampler.MoleculeDataSampler import (
    MoleculeFeatures,
)
import jax.numpy as jnp
import haiku as hk


class BackflowShift(lijax.Module):
    def __init__(
        self,
        name: str = "BackflowShift",
        n_hidden_bf_shift: list[int] = [40, 40],
    ):
        super().__init__(name=name)
        bf_shift_shape = n_hidden_bf_shift + [1]
        self.w_el_net_p = FCNN(bf_shift_shape, activation_fn=jax.nn.tanh, name="w_el")
        self.w_ion_net_p = FCNN(bf_shift_shape, activation_fn=jax.nn.tanh, name="w_ion")

    def init_modules(self):
        self.w_el_net = self.w_el_net_p.get_model()
        self.w_ion_net = self.w_ion_net_p.get_model()

    def __call__(
        self,
        embeddings,
        diff_el_el,
        dist_el_el,
        embeddings_el_el,
        diff_el_ion,
        dist_el_ion,
        embeddings_el_ions,
        mol: MoleculeFeatures,
    ) -> jax.Array:
        # basic feature
        shift_towards_electrons = calc_shift(
            embeddings,
            embeddings_el_el,
            self.w_el_net,
            diff_el_el,
            dist_el_el,
            feature_mask=mol.active_electrons,
            out_mask=mol.active_electrons,
            mask_self=True,
        )

        shift_towards_ions = calc_shift(
            embeddings,
            embeddings_el_ions,
            self.w_ion_net,
            diff_el_ion,
            dist_el_ion,
            feature_mask=mol.active_nuclears,
            out_mask=mol.active_electrons,
            mask_self=False,
        )

        scale_decay = hk.get_parameter(
            "scale_decay",
            shape=(),
            init=hk.initializers.Constant(0.5),
        )
        shift_decay = calculate_shift_decay(dist_el_ion, scale_decay, mol)
        shift = (shift_towards_electrons + shift_towards_ions) * shift_decay[:, None]
        scale_el = hk.get_parameter(
            "scale_el",
            shape=(),
            init=hk.initializers.Constant(-3.5),
        )
        shift = jnp.exp(scale_el) * shift

        return shift


def calc_shift(
    x, pair_embedding, net, diff, dist, feature_mask, out_mask, mask_self=True
):

    mask_non_identity = ~jnp.eye(x.shape[0], dtype=jnp.bool_)
    # x: [n_particles x n_features] -> [n_particles x n_particles2 x n_features]
    x = jnp.repeat(x[:, None, :], pair_embedding.shape[1], axis=1)
    if mask_self:
        x = x * mask_non_identity[:, :, None]
    features = jnp.concatenate([x, pair_embedding], axis=-1)
    # features: [n_particles x n_particles2 x (n_features+x_features)]

    shift = net(features)
    # dist: [n_particles x n_particles2]
    # shift: [n_particles x n_particles2 x 1]
    # diff: [n_particles x n_particles2 x 3]
    shift2 = (shift / (1 + dist[..., jnp.newaxis] ** 3)) * diff
    if mask_self:
        shift2 = shift2 * mask_non_identity[:, :, None]

    shift2 = shift2 * feature_mask[None, :, None]
    features_len = jnp.sum(feature_mask)
    out = jnp.sum(shift2, axis=-2) / features_len
    out = out * out_mask[:, None]
    # out:[n_particles x 3]
    return out


def calculate_shift_decay(d_el_ion, decaying_parameter, mol: MoleculeFeatures):
    """
    Computes the scaling factor ensuring that the contribution of the backflow shift decays in the proximity of a nucleus.

    Args:
        d_el_ion (array): Pairwise electron-ion distances
        Z (array): Nuclear charges
        decaying_parameter (array): Decaying parameters (same length as `Z`)

    Returns:
        array: Scalings for each electron

    """
    scale = decaying_parameter / mol.nuclear_charges
    a = (jnp.tanh((d_el_ion / scale) ** 2) * mol.active_nuclears[None, :]) + (
        mol.active_nuclears[None, :] == 0
    ) * mol.active_electrons[:, None]
    scaling = jnp.prod(a, axis=-1)
    return scaling
