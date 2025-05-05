import jax
from maml_vmc import lijax
from maml_vmc.models.utils.FCNN import FCNN
from maml_vmc.sampler.MoleculeDataSampler import (
    MoleculeFeatures,
)
import jax.numpy as jnp


class SchnetEmbedding(lijax.Module):
    def __init__(
        self,
        name: str = "SchnetEmbedding",
        emb_dim: int = 64,
        hidden_outputs_w: list[int] = [40, 40],
        hidden_outputs_h: list[int] = [40, 40],
        hidden_outputs_g: list[int] = [40],
        num_iterations: int = 2,
    ):

        super().__init__(name=name)
        self.num_iterations = num_iterations
        self.emb_dim = emb_dim
        shape_w = hidden_outputs_w + [emb_dim]
        shape_h = hidden_outputs_h + [emb_dim]
        shape_g = hidden_outputs_g + [emb_dim]
        self.ion_emb_net_p = FCNN(
            shape_h,
            activation_fn=jax.nn.tanh,
            use_last_layer_activation=True,
            name="ion_emb",
        )
        self.h_same_nets_p = [
            FCNN(
                shape_h,
                activation_fn=jax.nn.tanh,
                use_last_layer_activation=True,
                name=f"h_same_{i}",
            )
            for i in range(num_iterations)
        ]
        self.h_diff_nets_p = [
            FCNN(
                shape_h,
                activation_fn=jax.nn.tanh,
                use_last_layer_activation=True,
                name=f"h_diff_{i}",
            )
            for i in range(num_iterations)
        ]
        self.w_same_nets_p = [
            FCNN(shape_w, activation_fn=jax.nn.tanh, name=f"w_same_{i}")
            for i in range(num_iterations)
        ]
        self.w_diff_nets_p = [
            FCNN(shape_w, activation_fn=jax.nn.tanh, name=f"w_diff_{i}")
            for i in range(num_iterations)
        ]
        self.w_el_ion_nets_p = [
            FCNN(shape_w, activation_fn=jax.nn.tanh, name=f"w_el_ion_{i}")
            for i in range(num_iterations)
        ]
        self.g_func_nets_p = [
            FCNN(
                shape_g,
                activation_fn=jax.nn.tanh,
                use_last_layer_activation=True,
                name=f"g_func_{i}",
            )
            for i in range(num_iterations)
        ]

    def init_modules(self):
        self.ion_emb_net = self.ion_emb_net_p.get_model()
        self.h_same_nets = [h.get_model() for h in self.h_same_nets_p]
        self.h_diff_nets = [h.get_model() for h in self.h_diff_nets_p]
        self.w_same_nets = [h.get_model() for h in self.w_same_nets_p]
        self.w_diff_nets = [h.get_model() for h in self.w_diff_nets_p]
        self.w_el_ion_nets = [h.get_model() for h in self.w_el_ion_nets_p]
        self.g_func_nets = [h.get_model() for h in self.g_func_nets_p]

    def __call__(
        self, features_el_el, features_el_ion, mol: MoleculeFeatures
    ) -> jax.Array:
        # basic feature extraction
        num_active_electrons = mol.active_electrons.sum()
        num_active_ions = mol.active_nuclears.sum()
        mask_not_none = mol.active_electrons[:, None] * mol.active_electrons[None, :]
        mask_same = jnp.outer(mol.electron_spins, mol.electron_spins)
        mask_same = jnp.where(mask_same == 1, 1, 0)
        mask_diff = jnp.outer(mol.electron_spins, -1 * mol.electron_spins)
        mask_diff = jnp.where(mask_diff == 1, 1, 0)
        mask_non_identity = ~jnp.eye(features_el_el.shape[0], dtype=jnp.bool_)
        mask_same = mask_same & mask_non_identity & mask_not_none
        mask_diff = mask_diff & mask_non_identity & mask_not_none
        # add feature dimension
        Z = jnp.array(mol.nuclear_charges, jnp.float32)[:, None]
        ion_embeddings = self.ion_emb_net(Z) * mol.active_nuclears[:, None]

        # x: [n_electrons, emb_dim]
        x = jnp.ones(
            (features_el_el.shape[0], self.emb_dim), dtype=features_el_el.dtype
        )
        for i in range(self.num_iterations):

            h_same = self.h_same_nets[i](x)[None, ::].repeat(x.shape[0], axis=0)
            h_diff = self.h_diff_nets[i](x)[None, :, :].repeat(x.shape[0], axis=0)
            # h_same: [n_electrons, n_electrons, emb_dim]
            # h_diff: [n_electrons, n_electrons, emb_dim]
            w_same = self.w_same_nets[i](features_el_el)
            w_diff = self.w_diff_nets[i](features_el_el)
            emb_same = w_same * h_same * mask_same[:, :, None]
            emb_diff = w_diff * h_diff * mask_diff[:, :, None]
            embeddings_el_el = emb_same + emb_diff

            w_el_ions = (
                self.w_el_ion_nets[i](features_el_ion)
                * mol.active_electrons[:, None, None]
                * mol.active_nuclears[None, :, None]
            )
            embeddings_el_ions = w_el_ions * ion_embeddings[None, :, :]
            x = (jnp.sum(embeddings_el_el, axis=-2) / (num_active_electrons - 1)) + (
                jnp.sum(embeddings_el_ions, axis=-2) / (num_active_ions)
            )
            x = self.g_func_nets[i](x) * mol.active_electrons[:, None]

        return x, embeddings_el_el, embeddings_el_ions
