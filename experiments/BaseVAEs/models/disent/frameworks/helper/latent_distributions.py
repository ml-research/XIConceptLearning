from dataclasses import dataclass
from typing import Tuple, final

import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions import Normal, Distribution, Categorical

from disent.frameworks.helper.reductions import loss_reduction
from disent.util import TupleDataClass


# ========================================================================= #
# Helper Functions                                                          #
# ========================================================================= #


def kl_loss_direct(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # This is how the original VAE/BetaVAE papers do it:s
    # - we compute the kl divergence directly instead of approximating it
    return torch.distributions.kl_divergence(posterior, prior)


def kl_loss_approx(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None):
    # This is how pytorch-lightning-bolts does it:
    # See issue: https://github.com/PyTorchLightning/pytorch-lightning-bolts/issues/565
    # - we approximate the kl divergence instead of computing it analytically
    assert z_sampled is not None, 'to compute the approximate kl loss, z_sampled needs to be defined (cfg.kl_mode="approx")'
    return posterior.log_prob(z_sampled) - prior.log_prob(z_sampled)


_KL_LOSS_MODES = {
    'direct': kl_loss_direct,
    'approx': kl_loss_approx,
}


def kl_loss(posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None, mode='direct'):
    return _KL_LOSS_MODES[mode](posterior, prior, z_sampled)


# ========================================================================= #
# Vae Distributions                                                         #
# ========================================================================= #


class LatentDistribution(object):

    @dataclass
    class Params(TupleDataClass):
        """
        We use a params object so frameworks can check
        what kind of ops are supported, debug easier, and give type hints.
        - its a bit less efficient memory wise, but hardly...
        """

    def encoding_to_params(self, z_raw):
        raise NotImplementedError

    def params_to_representation(self, z_params: Params) -> torch.Tensor:
        raise NotImplementedError

    def params_to_distributions(self, z_params: Params) -> Tuple[Distribution, Distribution]:
        """
        make the posterior and prior distributions
        """
        raise NotImplementedError

    def params_to_distributions_and_sample(self, z_params: Params) -> Tuple[Tuple[Distribution, Distribution], torch.Tensor]:
        """
        Return the parameterized prior and the approximate posterior distributions,
        as well as a sample from the approximate posterior using the 'reparameterization trick'.
        """
        posterior, prior = self.params_to_distributions(z_params)
        # sample from posterior -- reparameterization trick!
        # ie. z ~ q(z|x)
        z_sampled = posterior.rsample()
        # return values
        return (posterior, prior), z_sampled

    @classmethod
    def compute_kl_loss(
            cls,
            posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None,
            mode: str = 'direct', reduction='batch_mean'
    ):
        """
        Compute the kl divergence
        """
        kl = kl_loss(posterior, prior, z_sampled, mode=mode)
        kl = loss_reduction(kl, reduction=reduction)
        return kl


# ========================================================================= #
# Normal Distribution                                                       #
# ========================================================================= #


class LatentDistributionNormal(LatentDistribution):
    """
    Latent distributions with:
    - posterior: normal distribution with diagonal covariance
    - prior: unit normal distribution
    """

    @dataclass
    class Params(LatentDistribution.Params):
        mean: torch.Tensor = None
        logvar: torch.Tensor = None

    @final
    def encoding_to_params(self, raw_z: Tuple[torch.Tensor, torch.Tensor]) -> Params:
        z_mean, z_logvar = raw_z
        return self.Params(z_mean, z_logvar)

    @final
    def params_to_representation(self, z_params: Params) -> torch.Tensor:
        return z_params.mean

    @final
    def params_to_distributions(self, z_params: Params) -> Tuple[Normal, Normal]:
        """
        Return the parameterized prior and the approximate posterior distributions.
        - The standard VAE parameterizes the gaussian normal with diagonal covariance.
        - logvar is used to avoid negative values for the standard deviation
        - Gaussian Encoder Model Distribution: pg. 25 in Variational Auto Encoders

        (✓) Visual inspection against reference implementations:
            https://github.com/google-research/disentanglement_lib (sample_from_latent_distribution)
            https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
        """
        z_mean, z_logvar = z_params
        # compute required values
        z_std = torch.exp(0.5 * z_logvar)
        # q: approximate posterior distribution
        posterior = Normal(z_mean, z_std)
        # p: prior distribution
        prior = Normal(torch.zeros_like(z_mean), torch.ones_like(z_std))
        # return values
        return posterior, prior

    @staticmethod
    def LEGACY_compute_kl_loss(mu, logvar, mode: str = 'direct', reduction='batch_mean'):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.
        FROM: https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (compute_gaussian_kl)
        """
        assert mode == 'direct', f'legacy reference implementation of KL loss only supports mode="direct", not {repr(mode)}'
        assert reduction == 'batch_mean', f'legacy reference implementation of KL loss only supports reduction="batch_mean", not {repr(reduction)}'
        # Calculate KL divergence
        kl_values = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # Sum KL divergence across latent vector for each sample
        kl_sums = torch.sum(kl_values, dim=1)
        # KL loss is mean of the KL divergence sums
        kl_loss = torch.mean(kl_sums)
        return kl_loss


# ========================================================================= #
# Categorical Distribution                                                       #
# ========================================================================= #

class LatentDistributionCategorical(LatentDistribution):
    """
    Latent distributions with:
    - posterior:
    - prior:
    """

    def __init__(self, temp: float, eps: float):
        self.temp = temp
        self.eps = eps

    # TODO: don't need this currently
    # def update_temperature(self, batch_idx: int):
    #     # Anneal the temperature at regular intervals
    #     if batch_idx % self.anneal_interval == 0:
    #         self.temp = np.maximum(self.temp * np.exp(- self.anneal_rate * batch_idx),
    #                                self.min_temp)

    @dataclass
    class Params(LatentDistribution.Params):
        """
        Parameters of a categorical distribution are the event probabilities of each category, we also store the logits
        that correspond to the unnormalized probabilities and for the sake of having multiple categorical distributions
        how many distributions (M, aka number of latent variables) and the number of categories, K.
        """
        M: torch.Tensor = None # number of latent variables
        K: torch.Tensor = None # number of categories
        logits: torch.Tensor = None # event logits
        probs: torch.Tensor = None # event probabilities, must sum to 1

    @final
    def encoding_to_params(self, raw_z: torch.Tensor) -> Params:
        assert len(raw_z.shape) == 3
        z = raw_z
        # Convert the categorical codes into probabilities
        z_p = F.softmax(z, dim=-1)
        return self.Params(z.shape[1], z.shape[2], z, z_p)

    @final
    def params_to_representation(self, z_params: Params) -> torch.Tensor:
        return z_params.probs.view(-1, z_params.M * z_params.K)

    @final
    def LEGACY_compute_kl_loss(self, probs, K, reduction='batch_mean'):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.
        FROM: https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (compute_gaussian_kl)
        """
        assert reduction == 'batch_mean', f'legacy reference implementation of KL loss only supports reduction="batch_mean", not {repr(reduction)}'
        # Convert the categorical codes into probabilities
        # # q_p = F.softmax(q, dim=-1)
        # Entropy of the logits
        h1 = probs * torch.log(probs + self.eps)
        # Cross entropy with the categorical distribution
        h2 = probs * np.log(1. / K + self.eps)
        kl_loss = torch.mean(torch.sum(h1 - h2, dim =(1,2)), dim=0)
        return kl_loss

    def params_to_distributions_and_sample(self, z_params: Params) -> Tuple[Tuple[Distribution, Distribution], torch.Tensor]:
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param z: (Tensor) Latent Codes [B x D x Q]
        :return: (Tensor) [B x D]
        """
        M, K, z_logits, z_probs = z_params

        # Sample from Gumbel
        u = torch.rand_like(z_logits)
        g = - torch.log(- torch.log(u + self.eps) + self.eps)

        # Gumbel-Softmax sample
        z_sample = F.softmax((z_logits + g) / self.temp, dim=-1)
        # print(self.temp)

        # just a sanity check
        # import numpy as np
        # print(np.round(z_sample[0].detach().cpu().numpy(), 3))

        # reshape for decoder
        z_sample = z_sample.view(-1, M * K)

        return z_probs, z_sample

# ========================================================================= #
# Factory                                                                   #
# ========================================================================= #


def make_latent_distribution(name: str, *kwargs) -> LatentDistribution:
    if name == 'normal':
        return LatentDistributionNormal()
    elif name == 'categorical':
        return LatentDistributionCategorical(*kwargs)
    else:
        raise KeyError(f'unknown vae distribution name: {name}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
