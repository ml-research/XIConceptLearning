from dataclasses import dataclass
from typing import Tuple, final

import torch
from torch.distributions import Distribution

from disent.frameworks.ae.unsupervised import AE
from disent.frameworks.helper.latent_distributions import make_latent_distribution, LatentDistribution


# ========================================================================= #
# framework_vae                                                             #
# ========================================================================= #


class CatVae(AE):
    """
    Variational Auto Encoder
    https://arxiv.org/abs/1312.6114
    """

    # override required z from AE
    REQUIRED_Z_MULTIPLIER = 1

    @dataclass
    class cfg(AE.cfg):
        beta: float = 1.
        latent_distribution: str = 'categorical'
        kl_loss_mode: str = 'direct'
        n_categories: int = 4
        n_variables: int = 2 # number of latent variables
        z_size: int = 8
        temp: float = 0.5
        eps: float = 1e-10

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # categorical distribution
        self._distributions: LatentDistribution = make_latent_distribution(self.cfg.latent_distribution,
                                                                           self.cfg.temp, self.cfg.eps,)

    # --------------------------------------------------------------------- #
    # VAE Training Step                                                     #
    # --------------------------------------------------------------------- #

    def compute_training_loss(self, batch, batch_idx):
        (x,), (x_targ,) = batch['x'], batch['x_targ']

        # TODO: don't need this currently
        # self._distributions.update_temperature(batch_idx)

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parameterizations
        z_params = self.training_encode_params(x)
        # sample from latent distribution
        z_probs, z_sampled = self.training_params_to_distributions_and_sample(z_params)
        # reconstruct without the final activation
        x_partial_recon = self.training_decode_partial(z_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        recon_loss = self.training_recon_loss(x_partial_recon, x_targ)  # E[log p(x|z)]
        # KL divergence & regularization
        kl_loss = self.training_kl_loss(z_probs, self.cfg.n_categories)  # D_kl(q(z|x) || p(z|x))
        # compute kl regularisation
        kl_reg_loss = self.training_regularize_kl(kl_loss)
        # compute combined loss
        loss = recon_loss + kl_reg_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': recon_loss,
            'kl_reg_loss': kl_reg_loss,
            'kl_loss': kl_loss,
            'elbo': -(recon_loss + kl_loss),
        }

    # --------------------------------------------------------------------- #
    # CatVAE - Overrides VAE                                                    #
    # --------------------------------------------------------------------- #

    @final
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get the deterministic latent representation (useful for visualisation)"""
        z_params = self.training_encode_params(x)
        z = self._distributions.params_to_representation(z_params)
        return z

    @final
    def training_encode_params(self, x: torch.Tensor) -> 'Params':
        """Get parametrisations of the latent distributions, which are sampled from during training."""
        z = self._model.encode(x)
        z = z.view(-1, self.cfg.n_variables, self.cfg.n_categories)
        z_params = self._distributions.encoding_to_params(z)
        return z_params

    @final
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Feed through the full deterministic model (useful for visualisation), also returns the latent representation
        """
        z_probs_flattened = self.encode(batch)
        z_probs = z_probs_flattened.view(-1, self.cfg.n_variables, self.cfg.n_categories)
        recon = self.decode(z_probs_flattened)
        return recon, z_probs

    # --------------------------------------------------------------------- #
    # CatVAE Model Utility Functions (Training)                                #
    # --------------------------------------------------------------------- #

    @final
    def training_params_to_distributions_and_sample(self, z_params: 'Params') -> Tuple[Tuple[Distribution, Distribution], torch.Tensor]:
        return self._distributions.params_to_distributions_and_sample(z_params)

    @final
    def training_params_to_distributions(self, z_params: 'Params') -> Tuple[Distribution, Distribution]:
        return self._distributions.params_to_distributions(z_params)

    @final
    def training_kl_loss(self, z_probs: torch.Tensor, K: int) -> torch.Tensor:
        return self._distributions.LEGACY_compute_kl_loss(
            z_probs, K,
            reduction=self.cfg.loss_reduction,
        )

    # --------------------------------------------------------------------- #
    # CatVAE Model Utility Functions (Overridable)                             #
    # --------------------------------------------------------------------- #

    def training_regularize_kl(self, kl_loss):
        if self.cfg.beta == 0:
            # numerical stability
            return torch.zeros_like(kl_loss)
        else:
            return self.cfg.beta * kl_loss


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
    net = CatVae(make_optimizer_fn=lambda params: Adam(params, lr=1e-3),
                     make_model_fn=lambda: AutoEncoder(
                         encoder=EncoderConv64(x_shape=(3, 64, 64), z_size=z_size),
                         decoder=DecoderConv64(x_shape=(3, 64, 64), z_size=z_size),
                     ),
                     cfg=CatVae.cfg(beta=1., n_categories=n_categories, n_variables=n_variables,
                                    temp=temp, eps=1e-12, z_size=z_size)
                 )
    # output = net(torch.randn(5, 3, 64, 64))
    X = torch.randn(5, 3, 64, 64)
    batch = {'x': (X,), 'x_targ': (X,)}
    loss = net.compute_training_loss(batch, batch_idx=0)
