from maml_vmc import lijax
from maml_vmc.models.deep_erwin.BackflowFactor import BackflowFactor
from maml_vmc.models.deep_erwin.BackflowShift import BackflowShift
from maml_vmc.models.deep_erwin.CASSCFBaseline import CASSCFBaseline
from maml_vmc.models.deep_erwin.CuspCorrection import CuspCorrection
from maml_vmc.models.deep_erwin.FeatureExtraction import FeatureExtraction
from maml_vmc.models.deep_erwin.JastrowFactor import JastrowFactor
from maml_vmc.models.deep_erwin.SchnetEmbedding import SchnetEmbedding
from maml_vmc.models.deep_erwin.utils import (
    DOWN,
    UP,
    get_distance_matrix_el_ion,
)
import jax.numpy as jnp

from maml_vmc.sampler.MoleculeDataSampler import (
    MoleculeFeatures,
)


class DeepErwin(lijax.Module):
    def __init__(
        self,
        backflow_factor_net: BackflowFactor,
        jastrow_factor_net: JastrowFactor = JastrowFactor(),
        backflow_shift_net: BackflowShift = BackflowShift(),
        feature_extractor: FeatureExtraction = FeatureExtraction(),
        embedding_net: SchnetEmbedding = SchnetEmbedding(),
        cusp_correction: CuspCorrection = CuspCorrection(),
        casscf_baseline=CASSCFBaseline(),
        name: str = "DeepErwin",
    ):
        self.feature_extractor_p = feature_extractor
        self.embedding_net_p = embedding_net
        self.backflow_shift_net_p = backflow_shift_net
        self.backflow_factor_net_p = backflow_factor_net
        self.jastrow_factor_net_p = jastrow_factor_net
        self.casscf_baseline_p = casscf_baseline
        self.cusp_correction = cusp_correction

        super().__init__(name=name)

    def init_modules(self):
        self.feature_extractor = self.feature_extractor_p.get_model()
        self.embedding_net = self.embedding_net_p.get_model()
        self.backflow_shift_net = self.backflow_shift_net_p.get_model()
        self.backflow_factor_net = self.backflow_factor_net_p.get_model()
        self.jastrow_factor_net = self.jastrow_factor_net_p.get_model()
        self.casscf_baseline = self.casscf_baseline_p.get_model()

    def __call__(self, mol: MoleculeFeatures) -> jnp.ndarray:
        if isinstance(mol, tuple):
            mol, debug = mol
        else:
            debug = False
        # basic feature extraction
        (
            features_el_el,
            features_el_ion,
            diff_el_el,
            dist_el_el,
            diff_el_ion,
            dist_el_ion,
        ) = self.feature_extractor(mol)
        # embedding
        # [link](maml_vmc/models/deep_erwin/SchnetEmbedding.py)
        embeddings, embeddings_el_el, embeddings_el_ions = self.embedding_net(
            features_el_el, features_el_ion, mol
        )
        # print(embeddings[2])
        # backflow shift
        # [link](maml_vmc/models/deep_erwin/BackflowShift.py)
        backflow_shift = self.backflow_shift_net(
            embeddings,
            diff_el_el,
            dist_el_el,
            embeddings_el_el,
            diff_el_ion,
            dist_el_ion,
            embeddings_el_ions,
            mol,
        )
        # apply the backflow shift
        shifted_electron_positions = (
            mol.electron_positions.reshape(-1, 3) + backflow_shift
        )
        diff_el_ion, dist_el_ion = get_distance_matrix_el_ion(
            mol.nuclear_positions, shifted_electron_positions
        )

        # get baseline Phi for the determinant
        # [link](maml_vmc/models/deep_erwin/CASSCFBaseline.py)
        phi_mo_matrix_up, phi_mo_matrix_down, det_weights = self.casscf_baseline(
            diff_el_ion, dist_el_ion, mol
        )

        # get backflow factor
        # [link](maml_vmc/models/deep_erwin/BackflowFactor.py)
        backflow_factor_up, backflow_factor_dn = self.backflow_factor_net(
            embeddings, mol
        )

        # apply the backflow factor
        phi_mo_matrix_up2 = phi_mo_matrix_up * backflow_factor_up
        phi_mo_matrix_down2 = phi_mo_matrix_down * backflow_factor_dn

        phi_mo_matrix_up3, phi_mo_matrix_down3 = mask_matrix(
            phi_mo_matrix_up2, phi_mo_matrix_down2, mol
        )

        log_psi_sqr = _evaluate_sum_of_determinants(
            phi_mo_matrix_up3, phi_mo_matrix_down3, det_weights
        )

        # get jastrow factor
        # [link](maml_vmc/models/deep_erwin/JastrowFactor.py)
        jastrow_factor = self.jastrow_factor_net(embeddings, mol)
        # apply the jastrow factor
        log_psi_sqr = log_psi_sqr + jastrow_factor

        # apply electron-electron cusp correction
        cc = self.cusp_correction(dist_el_el, mol)
        log_psi_sqr = log_psi_sqr + cc

        return log_psi_sqr


def _evaluate_sum_of_determinants(mo_matrix_up, mo_matrix_dn, ci_weights):
    LOG_EPSILON = 1e-8
    sign_up, log_up = jnp.linalg.slogdet(mo_matrix_up)
    sign_dn, log_dn = jnp.linalg.slogdet(mo_matrix_dn)
    log_total = log_up + log_dn
    sign_total = sign_up * sign_dn
    # print(log_total)
    # exit()
    log_shift = jnp.max(log_total, axis=-1, keepdims=True)
    psi = jnp.exp(log_total - log_shift) * sign_total
    psi2 = jnp.sum(psi * ci_weights, axis=-1)  # sum over determinants
    log_psi_sqr = 2 * (
        jnp.log(jnp.abs(psi2) + LOG_EPSILON) + jnp.squeeze(log_shift, -1)
    )
    return log_psi_sqr


def mask_matrix(mo_matrix_up, mo_matrix_dn, mol: MoleculeFeatures):

    up_mask = mol.electron_spins == UP

    up_mask = jnp.outer(up_mask, up_mask)[None]
    eye = jnp.eye(mo_matrix_up.shape[1], dtype=mo_matrix_up.dtype)[None, :, :]
    man_up_eye = (mol.electron_spins != UP)[None, :, None] * eye
    mo_matrix_up = (mo_matrix_up * up_mask) + man_up_eye

    dn_mask = mol.electron_spins == DOWN
    dn_mask = jnp.roll(dn_mask, -mol.n_up)
    dn_mask = jnp.outer(dn_mask, dn_mask)[None]
    man_dn_eye = jnp.roll((mol.electron_spins != DOWN), -mol.n_up)[None, :, None] * eye
    mo_matrix_dn = (mo_matrix_dn * dn_mask) + man_dn_eye

    return mo_matrix_up, mo_matrix_dn
