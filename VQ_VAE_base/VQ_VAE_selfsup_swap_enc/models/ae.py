import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.modules as modules


def catch_nan(x):
	if torch.isnan(x):
		return 0.
	else:
		return x


# TODO: add passing unsimilar number of prototypes per group
class AE(nn.Module):
	def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, enc_size=6):
		super(AE, self).__init__()

		self._encoder = modules.Encoder(3, num_hiddens,
		                        num_residual_layers,
		                        num_residual_hiddens)
		self.latent_shape = tuple(self._encoder(torch.rand(1, 3, 64, 64)).shape[1:])
		self.latent_flat = np.prod(self.latent_shape)

		self._decoder_z = modules.Decoder(num_hiddens,
		                        num_hiddens,
		                        num_residual_layers,
		                        num_residual_hiddens)
		self._encoder_linear = nn.Sequential(
			nn.Linear(self.latent_flat, enc_size),
			nn.BatchNorm1d(enc_size),
			nn.ReLU(),
		)
		self._decoder_linear = nn.Sequential(
			nn.Linear(enc_size, self.latent_flat),
			nn.BatchNorm1d(self.latent_flat),
			nn.ReLU(),
		)

	def forward(self, imgs):
		(x0, x0_a, x1) = imgs # sample, augmented sample, negative sample

		z0 = self._encoder(x0) #[B, F, W, H]
		z0_a = self._encoder(x0_a) #[B, F, W, H]
		z1 = self._encoder(x1) #[B, F, W, H]

		z0 = self._encoder_linear(z0.view(-1, self.latent_flat))
		z0_a = self._encoder_linear(z0_a.view(-1, self.latent_flat))
		z1 = self._encoder_linear(z1.view(-1, self.latent_flat))

		# reconstruct z
		z0 = self._decoder_linear(z0)
		z0_a = self._decoder_linear(z0_a)
		z1 = self._decoder_linear(z1)

		z0_recon = self._decoder_z(z0.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2])) # [B, 3, W, H]
		z0_a_recon = self._decoder_z(z0_a.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2])) # [B, 3, W, H]
		z1_recon = self._decoder_z(z1.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2])) # [B, 3, W, H]

		return (z0_recon, z0_a_recon, z1_recon)


if __name__ == '__main__':
	batch_size = 256

	num_hiddens = 64
	num_residual_hiddens = 32
	num_residual_layers = 2

	learning_rate = 1e-3

	model = AE(num_hiddens=num_hiddens, num_residual_layers=num_residual_layers,
	                     num_residual_hiddens=num_residual_hiddens).to('cpu')

	data = torch.rand(15, 3, 64, 64)
	data_recon = model.forward(data)
