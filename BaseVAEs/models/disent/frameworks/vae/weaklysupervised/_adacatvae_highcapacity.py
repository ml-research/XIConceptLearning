import torch
from dataclasses import dataclass

from torch.distributions import Distribution
from torch.nn import functional as F

from disent.frameworks.vae.unsupervised import CatVae


# ========================================================================= #
# Ada-GVAE                                                                  #
# ========================================================================= #


class AdaCatVae_HighCapacity(CatVae):

    """
    Weakly Supervised Disentanglement Learning Without Compromises: https://arxiv.org/abs/2002.02886
    - pretty much a beta-vae with averaging between decoder outputs to form weak supervision signal.
    - GAdaVAE:   Averaging from https://arxiv.org/abs/1809.02383
    - ML-AdaVAE: Averaging from https://arxiv.org/abs/1705.08841

    MODIFICATION:
    - Symmetric KL Calculation used by default, described in: https://openreview.net/pdf?id=8VXvj1QNRl1
    """

    @dataclass
    class cfg(CatVae.cfg):
        average_mode: str = 'gvae'
        symmetric_kl: bool = True
        n_groups: int = 3
        n_protos: int = 6

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # averaging modes
        self._compute_average_fn = {
            'gvae': compute_average_gvae,
            'ml-vae': compute_average_ml_vae
        }[self.cfg.average_mode]

    def compute_training_loss(self, batch, batch_idx):
        """
        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (GroupVAEBase & MLVae)
            - only difference for GroupVAEBase & MLVae how the mean parameterisations are calculated
        """
        # (x0, x1), (x0_targ, x1_targ) = batch['x'], batch['x_targ']
        (x0, x1), (x0_targ, x1_targ), share_mask = batch['x'], batch['x_targ'], batch['shared_mask']

        assert x0.shape == x1.shape

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parameterizations
        z0_params = self.training_encode_params(x0)
        z1_params = self.training_encode_params(x1)

        assert z0_params.K == z1_params.K == self.cfg.n_categories
        assert z0_params.M == z1_params.M == self.cfg.n_variables

        # intercept and mutate z [SPECIFIC TO ADAVAE]
        (z0_params, z1_params), intercept_logs = self.intercept_z(all_params=(z0_params, z1_params),
                                                                  share_mask=share_mask)
        # sample from latent distribution
        z0_probs, z0_sampled = self.training_params_to_distributions_and_sample(z0_params)
        z1_probs, z1_sampled = self.training_params_to_distributions_and_sample(z1_params)
        # reconstruct without the final activation
        x0_partial_recon = self.training_decode_partial(z0_sampled)
        x1_partial_recon = self.training_decode_partial(z1_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        recon0_loss = self.training_recon_loss(x0_partial_recon, x0_targ)  # E[log p(x|z)]
        recon1_loss = self.training_recon_loss(x1_partial_recon, x1_targ)  # E[log p(x|z)]
        ave_recon_loss = (recon0_loss + recon1_loss) / 2
        # KL divergence
        kl0_loss = self.training_kl_loss(z0_probs, z0_params.K)  # D_kl(q(z|x) || p(z|x), d0_prior)
        kl1_loss = self.training_kl_loss(z1_probs, z1_params.K)  # D_kl(q(z|x) || p(z|x), d1_prior)
        ave_kl_loss = (kl0_loss + kl1_loss) / 2
        # compute kl regularisation
        ave_kl_reg_loss = self.training_regularize_kl(ave_kl_loss)
        # compute combined loss - must be same as the BetaVAE
        loss = ave_recon_loss + ave_kl_reg_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': ave_recon_loss,
            'kl_reg_loss': ave_kl_reg_loss,
            'kl_loss': ave_kl_loss,
            'elbo': -(ave_recon_loss + ave_kl_loss),
            **intercept_logs,
        }

    def intercept_z(self, all_params, share_mask=None):
        """
        Adaptive VAE Glue Method, putting the various components together
        1. find differences between deltas
        2. estimate a threshold for differences
        3. compute a shared mask from this threshold
        4. average together elements that should be considered shared

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (aggregate_argmax)
        """
        z0_params, z1_params = all_params
        # compute the deltas
        z_deltas = self.compute_kl_deltas(z0_params.probs, z1_params.probs, symmetric_kl=self.cfg.symmetric_kl)

        # shared elements that need to be averaged, computed per pair in the batch.
        if share_mask is None:
            share_mask = self.compute_shared_mask(z_deltas)

        # compute average posteriors
        new_args = self.compute_averaged(z0_params, z1_params, share_mask, compute_average_fn=self._compute_average_fn)
        # return new args & generate logs
        return new_args, {'shared': share_mask.sum(dim=1).float().mean()}

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # HELPER                                                                #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @classmethod
    def compute_kl_deltas(cls, z0_probs: torch.Tensor, z1_probs: torch.Tensor, symmetric_kl: bool):
        """
        (✓) Visual inspection against reference implementation
        https://github.com/google-research/disentanglement_lib (compute_kl)
        - difference is that they don't multiply by 0.5 to get true kl, but that's not needed

        TODO: this might be numerically unstable with f32 passed to distributions
        """
        # shared elements that need to be averaged, computed per pair in the batch.
        # [𝛿_i ...]
        if symmetric_kl:
            # FROM: https://openreview.net/pdf?id=8VXvj1QNRl1
            kl_deltas_A = cls._kld_cat_elem_pairs(z0_probs, z1_probs)
            kl_deltas_B = cls._kld_cat_elem_pairs(z1_probs, z0_probs)
            kl_deltas = (0.5 * kl_deltas_A) + (0.5 * kl_deltas_B)
        else:
            kl_deltas = cls._kld_cat_elem_pairs(z0_probs, z1_probs)
        # return values
        return kl_deltas

    @classmethod
    def _kld_cat_elem_pairs(cls, z0_probs: torch.Tensor, z1_probs: torch.Tensor):
        """
        See https://stats.stackexchange.com/questions/72611/kl-divergence-between-two-categorical-multinomial-distributions-gives-negative-v
        :param z0_probs:
        :param z1_probs:
        :return:
        """
        # Convert the categorical codes into probabilities
        # q_0_p = F.softmax(q_0, dim=-1)
        # q_1_p = F.softmax(q_1, dim=-1)
        kld_coord = torch.sum(z0_probs * torch.log(z0_probs / z1_probs), dim=2)  # kld per variable coordinate
        return kld_coord

    @classmethod
    def compute_shared_mask(cls, z_deltas):
        """
        Core of the adaptive VAE algorithm, estimating which factors
        have changed (or in this case which are shared and should remained unchanged
        by being be averaged) between pairs of observations.

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (aggregate_argmax)
            - Implementation conversion is non-trivial, items are histogram binned.
              If we are in the second histogram bin, ie. 1, then kl_deltas <= kl_threshs
            - TODO: (aggregate_labels) An alternative mode exists where you can bind the
                    latent variables to any individual label, by one-hot encoding which
                    latent variable should not be shared: "enforce that each dimension
                    of the latent code learns one factor (dimension 1 learns factor 1)
                    and enforce that each factor of variation is encoded in a single
                    dimension."
        """
        # threshold τ
        z_threshs = cls.estimate_threshold(z_deltas)
        # true if 'unchanged' and should be average
        shared_mask = z_deltas < z_threshs
        # return
        return shared_mask


    @classmethod
    def estimate_threshold(cls, kl_deltas, keepdim=True):
        """
        Compute the threshold for each image pair in a batch of kl divergences of all elements of the latent distributions.
        It should be noted that for a perfectly trained model, this threshold is always correct.

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (aggregate_argmax)
        """
        maximums = kl_deltas.max(axis=1, keepdim=keepdim).values
        minimums = kl_deltas.min(axis=1, keepdim=keepdim).values
        return (0.5 * minimums) + (0.5 * maximums)

    @classmethod
    def compute_averaged(cls, z0_params, z1_params, share_mask, compute_average_fn: callable):
        def where_by_latent_variable_id(share_mask, ave_logits, single_logits):
            # TODO: double check if gradient is passed through clone
            tmp = torch.clone(ave_logits)
            # for those variable coordinates which are not shared, use thes logits of the single data sample,
            # otherwise use the averaged logits of the paired data samples
            tmp[~share_mask] = single_logits[~share_mask]
            return tmp

        # update the boolean share mask to high capacity case for proper averaging
        update_share_mask = update_share_mask_to_highcapacity(share_mask, cls.cfg.n_groups, cls.cfg.n_protos)

        M, K, z0_logits, z0_probs = z0_params
        M, K, z1_logits, z1_probs = z1_params

        # compute average posteriors
        ave_logits = compute_average_fn(
            z0_logits, z1_logits,
        )
        # select averages
        ave_z0_logits = where_by_latent_variable_id(update_share_mask, ave_logits, z0_logits)
        ave_z1_logits = where_by_latent_variable_id(update_share_mask, ave_logits, z1_logits)

        # update event probabilities based on average logits
        z0_probs = F.softmax(ave_z0_logits, dim=-1)
        z1_probs = F.softmax(ave_z1_logits, dim=-1)

        # return values
        return z0_params.__class__(M, K, ave_z0_logits, z0_probs), z1_params.__class__(M, K, ave_z1_logits, z1_probs)


# ========================================================================= #
# Averaging Functions                                                       #
# ========================================================================= #

def update_share_mask_to_highcapacity(share_mask, n_groups, n_protos):
    """
    Update the share mask that only contains boolean values for the number of groups to the high capacity case in which
    each proto class has its own variable.
    :param share_mask:
    :param n_groups:
    :param n_protos:
    :return:
    """
    update_share_mask = torch.zeros((share_mask.shape[0], n_groups*n_protos), dtype=torch.bool,
                                    device=share_mask.device)
    for group_id in range(n_groups):
        update_share_mask[:, (group_id * n_protos):((group_id+1) * n_protos)] = \
            share_mask[:, group_id].unsqueeze(dim=1).repeat(1, n_protos)
    return update_share_mask


def compute_average_gvae(z0_logits, z1_logits):
    """
    Compute the arithmetic mean of the encoder distributions.
    - Ada-GVAE Averaging function

    (✓) Visual inspection against reference implementation:
        https://github.com/google-research/disentanglement_lib (GroupVAEBase.model_fn)
    """
    ave_logits = 1 / 2 * (z0_logits + z1_logits)
    return ave_logits

def compute_average_ml_vae(z0_mean, z0_logvar, z1_mean, z1_logvar):
    """
    Compute the product of the encoder distributions.
    - Ada-ML-VAE Averaging function

    (✓) Visual inspection against reference implementation:
        https://github.com/google-research/disentanglement_lib (MLVae.model_fn)

    # TODO: recheck
    """
    raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

from torch.optim import Adam
from disent.model.ae import EncoderConv64, DecoderConv64, AutoEncoder

if __name__ == '__main__':
    n_categories = 4
    n_variables = 2
    z_size = n_categories * n_variables
    temp = 0.5
    net = AdaCatVae(make_optimizer_fn=lambda params: Adam(params, lr=1e-3),
                     make_model_fn=lambda: AutoEncoder(
                         encoder=EncoderConv64(x_shape=(3, 64, 64), z_size=z_size),
                         decoder=DecoderConv64(x_shape=(3, 64, 64), z_size=z_size),
                     ),
                     cfg=AdaCatVae.cfg(beta=1., n_categories=n_categories, n_variables=n_variables,
                                    temp=temp, eps=1e-12, z_size=z_size,
                                    average_mode='gvae', symmetric_kl=False)
                 )
    # output = net(torch.randn(5, 3, 64, 64))
    X = (torch.randn(5, 3, 64, 64), torch.randn(5, 3, 64, 64))
    batch = {'x': X, 'x_targ': X}
    loss = net.compute_training_loss(batch, batch_idx=0)
