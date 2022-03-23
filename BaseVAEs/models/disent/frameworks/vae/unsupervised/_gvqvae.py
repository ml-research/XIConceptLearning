from dataclasses import dataclass
from typing import Tuple, final

import numpy as np
import torch
from torch.distributions import Distribution

import disent.frameworks.helper.proto_modules as proto_modules
from disent.frameworks.ae.unsupervised import AE
from disent.frameworks.helper.latent_distributions import make_latent_distribution, LatentDistribution


# ========================================================================= #
# framework_vae                                                             #
# ========================================================================= #


class GVQVae(AE):
    """
    Group Vector Quantization Variational Auto Encoder
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
        z_size: int = 24
        temp: float = 0.5
        eps: float = 1e-10
        proto_dim: int = 12
        device: str = 'cuda'
        agg_type: str = 'linear'

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        # required_z_multiplier
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # vector quantization layer
        self.proto_layer = proto_modules.VectorQuantizationLayer(
            beta=self.cfg.beta,
            input_dim=self.cfg.proto_dim,
            n_proto_vecs=self.cfg.n_categories * np.ones(self.cfg.n_variables, dtype=np.int),
            device=self.cfg.device
        )
        # self.proto_agg_layer = proto_modules.ProtoAggregateLayer(n_protos=self.cfg.n_variables,
        #                                                    dim_protos=self.cfg.proto_dim,
        #                                                    layer_type=self.cfg.agg_type)

    # --------------------------------------------------------------------- #
    # GVQVAE Training Step                                                     #
    # --------------------------------------------------------------------- #

    def compute_training_loss(self, batch, batch_idx):
        (x,), (x_targ,) = batch['x'], batch['x_targ']

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parameterizations
        encoding = self.encode(x)
        # reshape to group form : [batch, latent] --> [batch, n_groups, n_categories]
        encoding = encoding.view(-1, self.cfg.n_variables, self.cfg.proto_dim)

        # compute the distance of the encoding 'slot' of each training example to the prototype of each group and
        # create vector quntized code
        quantized_encodings, vq_loss, _ = self.proto_layer(encoding) # [batch, n_group, n_proto]

        # TODO: aggregation is linear layer currently --> should decoder be permutation invariant?
        # aggregate the quantized codes from each group
        # quantized_encodings_agg = self.proto_agg_layer(quantized_encodings)
        quantized_encodings_agg = quantized_encodings.view(-1, self.cfg.n_variables * self.cfg.proto_dim)

        # decode the aggregated version
        recon = self.decode(quantized_encodings_agg)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        recon_loss = self.training_recon_loss(recon, x_targ)  # E[log p(x|z)]
        # # KL divergence & regularization
        # kl_loss = self.training_kl_loss(d_posterior, d_prior, z_sampled)  # D_kl(q(z|x) || p(z|x))
        # # compute kl regularisation
        # kl_reg_loss = self.training_regularize_kl(kl_loss)
        # compute combined loss
        loss = recon_loss + vq_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            # 'elbo': -(recon_loss + kl_loss),
        }

    # --------------------------------------------------------------------- #
    # VAE - Overrides AE                                                    #
    # --------------------------------------------------------------------- #

    @final
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get the deterministic latent representation (useful for visualisation)"""
        return self._model.encode(x)

    @final
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z into reconstruction x_recon (useful for visualisation)"""
        return self._recons.activate(self._model.decode(z))

    @final
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed through the full deterministic model (useful for visualisation), also returns the latent representation
        """
        # latent distribution parameterizations
        encoding = self.encode(x)
        # reshape to group form : [batch, latent] --> [batch, n_groups, n_categories]
        encoding = encoding.view(-1, self.cfg.n_variables, self.cfg.proto_dim)

        # compute the distance of the encoding 'slot' of each training example to the prototype of each group and
        # create vector quntized code
        quantized_encodings, vq_loss, dists = self.proto_layer(encoding) # [batch, n_group, n_proto]

        # aggregate the quantized codes from each group
        # quantized_encodings_agg = self.proto_agg_layer(quantized_encodings)
        quantized_encodings_agg = quantized_encodings.view(-1, self.cfg.n_variables * self.cfg.proto_dim)

        # decode the aggregated version
        recon = self.decode(quantized_encodings_agg)
        return recon, dists
    # --------------------------------------------------------------------- #
    # VAE Model Utility Functions (Training)                                #
    # --------------------------------------------------------------------- #

    @final
    def training_params_to_distributions_and_sample(self, z_params: 'Params') -> Tuple[Tuple[Distribution, Distribution], torch.Tensor]:
        return self._distributions.params_to_distributions_and_sample(z_params)

    @final
    def training_params_to_distributions(self, z_params: 'Params') -> Tuple[Distribution, Distribution]:
        return self._distributions.params_to_distributions(z_params)

    @final
    def training_kl_loss(self, d_posterior: Distribution, d_prior: Distribution, z_sampled: torch.Tensor = None) -> torch.Tensor:
        return self._distributions.compute_kl_loss(
            d_posterior, d_prior, z_sampled,
            mode=self.cfg.kl_loss_mode,
            reduction=self.cfg.loss_reduction,
        )

    # --------------------------------------------------------------------- #
    # VAE Model Utility Functions (Overridable)                             #
    # --------------------------------------------------------------------- #

    def training_regularize_kl(self, kl_loss):
        return kl_loss


# TODO: reference VQVAE code:
# class VectorQuantizer(nn.Module):
#     """
#     Reference:
#     [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
#     """
#     def __init__(self,
#                  num_embeddings: int,
#                  embedding_dim: int,
#                  beta: float = 0.25):
#         super(VectorQuantizer, self).__init__()
#         self.K = num_embeddings
#         self.D = embedding_dim
#         self.beta = beta
#
#         self.embedding = nn.Embedding(self.K, self.D)
#         self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
#
#     def forward(self, latents: Tensor) -> Tensor:
#         latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
#         latents_shape = latents.shape
#         flat_latents = latents.view(-1, self.D)  # [BHW x D]
#
#         # Compute L2 distance between latents and embedding weights
#         dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
#                torch.sum(self.embedding.weight ** 2, dim=1) - \
#                2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]
#
#         # Get the encoding that has the min distance
#         encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]
#
#         # Convert to one-hot encodings
#         device = latents.device
#         encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
#         encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]
#
#         # Quantize the latents
#         quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
#         quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]
#
#         # Compute the VQ Losses
#         commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
#         embedding_loss = F.mse_loss(quantized_latents, latents.detach())
#
#         vq_loss = commitment_loss * self.beta + embedding_loss
#
#         # Add the residue back to the latents
#         quantized_latents = latents + (quantized_latents - latents).detach()
#
#         return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]



# ========================================================================= #
# END                                                                       #
# ========================================================================= #

from torch.optim import Adam
from disent.model.ae import EncoderConv64, DecoderConv64, AutoEncoder

if __name__ == '__main__':
    n_categories = 4
    n_variables = 2
    proto_dim = 12
    z_size = proto_dim * n_variables
    temp = 0.5
    net = GVQVae(make_optimizer_fn=lambda params: Adam(params, lr=1e-3),
                     make_model_fn=lambda: AutoEncoder(
                         encoder=EncoderConv64(x_shape=(3, 64, 64), z_size=z_size),
                         decoder=DecoderConv64(x_shape=(3, 64, 64), z_size=z_size),
                     ),
                     cfg=GVQVae.cfg(beta=0.25, n_categories=n_categories, n_variables=n_variables,
                                    temp=temp, eps=1e-12, z_size=z_size, proto_dim=proto_dim, device='cpu',
                                    agg_type='linear')
                 )
    # output = net(torch.randn(5, 3, 64, 64))
    X = torch.randn(5, 3, 64, 64)
    batch = {'x': (X,), 'x_targ': (X,)}
    loss = net.compute_training_loss(batch, batch_idx=0)
