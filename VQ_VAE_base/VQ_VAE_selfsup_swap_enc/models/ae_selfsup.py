import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.modules_proto as proto_modules
import models.modules as modules


def catch_nan(x):
	if torch.isnan(x):
		return 0.
	else:
		return x


# TODO: add passing unsimilar number of prototypes per group
class AE_selfsup(nn.Module):
	def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, enc_size,
	             num_groups, num_protos, agg_type='linear', device='cpu'):
		super(AE_selfsup, self).__init__()

		# add 1 encoding group that should contain all the information that the examples should not share, i.e.
		# position, rotation, augmentation, etc
		self.num_groups = num_groups + 1
		self.num_proto_groups = num_groups
		self.num_protos = num_protos
		self.device = device
		self.agg_type = agg_type

		self._encoder = modules.Encoder(3, num_hiddens,
		                        num_residual_layers,
		                        num_residual_hiddens)
		self._decoder_z = modules.Decoder(num_hiddens,
		                        num_hiddens,
		                        num_residual_layers,
		                        num_residual_hiddens)

		self.latent_shape = tuple(self._encoder(torch.rand(1, 3, 64, 64)).shape[1:])
		self.latent_flat = np.prod(self.latent_shape)

		self._encoder_linear = nn.Sequential(
			nn.Linear(self.latent_flat, enc_size),
			nn.BatchNorm1d(enc_size),
			nn.ReLU(),
		)

		# latent space which contains all information and should be split into variables
		self.latent_shape_2 = self._encoder_linear(self._encoder(torch.rand(2, 3, 64, 64)).view(-1, self.latent_flat)
		                                           ).shape[1]
		# self.proto_dim = int(self.latent_shape_2 / self.num_groups)
		self.proto_dim = 256
		self.latent_shape_3 = self.proto_dim * self.num_groups

		# each group should have its own mlp head
		self.split_linears = nn.ModuleList([nn.Sequential(
			nn.Linear(self.latent_shape_2, self.proto_dim),
			nn.ReLU(),
		) for i in range(self.num_groups)])

		self._decoder_linear = nn.Sequential(
			nn.Linear(self.latent_shape_3, self.latent_flat),
			nn.BatchNorm1d(self.latent_flat),
			nn.ReLU(),
		)

		# TODO: just for testing now
		self.embeddings = dict()
		for group_id in range(num_groups):
			self.embeddings[group_id] = nn.Embedding(4, self.proto_dim).to(self.device)
			self.embeddings[group_id].weight.data.uniform_(-1 / 4, 1 / 4)

	def split(self, z):
		# z: [B, D]
		# iterate over each mlp head and combine to one tensor
		# [B, G, D_P], D_P = D/G
		return torch.stack([self.split_linears[i].forward(z) for i in range(self.num_groups)]).permute(1, 0, 2)

	def forward(self, imgs, k=None):
		(x0, x0_a, x1) = imgs # sample, augmented sample, negative sample

		# [B, F, W, H]
		z0 = self._encoder(x0)
		z0_a = self._encoder(x0_a)
		z1 = self._encoder(x1)

		# [B, D]
		z0 = self._encoder_linear(z0.view(-1, self.latent_flat))
		z0_a = self._encoder_linear(z0_a.view(-1, self.latent_flat))
		z1 = self._encoder_linear(z1.view(-1, self.latent_flat))

		# TODO: add attention over feature space rather than linear?
		# [B, D] --> [B, G, D_P], D_P = D/G
		z0 = self.split(z0)
		z0_a = self.split(z0_a)
		z1 = self.split(z1)

		# estimate which attributes are shared
		shared_masks, distances = self._comp_shared_group_ids(z0, z0_a, z1, k)

		triplet_loss, distances_emb = self._compute_triplet_loss(z0, z0_a, z1, shared_masks)

		# swap encodings to be shared
		z0, z0_a, z1, distances_emb = self._swap_encodings(z0, z0_a, z1, shared_masks)

		# reconstruct z
		# [B, G, D_P] --> [B, D]
		z0 = self._decoder_linear(z0.reshape(-1, self.latent_shape_3))
		z0_a = self._decoder_linear(z0_a.reshape(-1, self.latent_shape_3))
		z1 = self._decoder_linear(z1.reshape(-1, self.latent_shape_3))

		# [B, D] --> [B, 3, W, H]
		z0_recon = self._decoder_z(z0.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]))
		z0_a_recon = self._decoder_z(z0_a.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]))
		z1_recon = self._decoder_z(z1.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]))

		return (z0_recon, z0_a_recon, z1_recon), triplet_loss, distances, distances_emb

	def sanity_forward(self, imgs):
		(x0, x0_a, x1) = imgs # sample, augmented sample, negative sample

		# [B, F, W, H]
		z0 = self._encoder(x0)
		# z0_a = self._encoder(x0_a)
		z1 = self._encoder(x1)

		# [B, D]
		z0 = self._encoder_linear(z0.view(-1, self.latent_flat))
		# z0_a = self._encoder_linear(z0_a.view(-1, self.latent_flat))
		z1 = self._encoder_linear(z1.view(-1, self.latent_flat))

		# TODO: add attention over feature space rather than linear?
		# [B, D] --> [B, G, D_P], D_P = D/G
		z0 = self.split(z0)
		# z0_a = self.split(z0_a)
		z1 = self.split(z1).view(-1, self.num_groups, self.proto_dim)

		z0_swap0 = torch.clone(z0)
		# z0_a_swap0 = torch.clone(z0_a)
		z1_swap0 = torch.clone(z1)

		z0_swap1 = torch.clone(z0)
		# z0_a_swap1 = torch.clone(z0_a)
		z1_swap1 = torch.clone(z1)

		z0_swap0[:, 0] = z1[:, 0]
		z1_swap0[:, 0] = z0[:, 0]
		z0_swap1[:, 1] = z1[:, 1]
		z1_swap1[:, 1] = z0[:, 1]


		# reconstruct z
		# [B, G, D_P] --> [B, D]
		z0 = self._decoder_linear(z0.reshape(-1, self.latent_shape_3))
		# z0_a = self._decoder_linear(z0_a.reshape(-1, self.latent_shape_3))
		z1 = self._decoder_linear(z1.reshape(-1, self.latent_shape_3))
		z0_swap0 = self._decoder_linear(z0_swap0.reshape(-1, self.latent_shape_3))
		# z0_a = self._decoder_linear(z0_a.reshape(-1, self.latent_shape_3))
		z1_swap0 = self._decoder_linear(z1_swap0.reshape(-1, self.latent_shape_3))
		z0_swap1 = self._decoder_linear(z0_swap1.reshape(-1, self.latent_shape_3))
		# z0_a = self._decoder_linear(z0_a.reshape(-1, self.latent_shape_3))
		z1_swap1 = self._decoder_linear(z1_swap1.reshape(-1, self.latent_shape_3))

		# [B, D] --> [B, 3, W, H]
		z0_recon = self._decoder_z(z0.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]))
		# z0_a_recon = self._decoder_z(z0_a.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]))
		z1_recon = self._decoder_z(z1.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]))
		z0_recon_swap0 = self._decoder_z(z0_swap0.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]))
		# z0_a_recon = self._decoder_z(z0_a.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]))
		z1_recon_swap0 = self._decoder_z(z1_swap0.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]))
		z0_recon_swap1 = self._decoder_z(z0_swap1.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]))
		# z0_a_recon = self._decoder_z(z0_a.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]))
		z1_recon_swap1 = self._decoder_z(z1_swap1.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]))

		return z0_recon, z1_recon, z0_recon_swap0, z1_recon_swap0, z0_recon_swap1, z1_recon_swap1

	def _swap_encodings(self, inputs0, inputs0_a, inputs1, shared_masks):
		# shared_masks: [G, B]
		# [B, G, L] --> [G, B, L]
		inputs0 = inputs0.permute(1, 0, 2)
		inputs0_a = inputs0_a.permute(1, 0, 2)
		inputs1 = inputs1.permute(1, 0, 2)

		inputs0_swap = torch.clone(inputs0)
		inputs0_a_swap = torch.clone(inputs0_a)
		inputs1_swap = torch.clone(inputs1)

		distances0_emb = []
		distances0_a_emb = []
		distances1_emb = []

		for group_id in range(self.num_proto_groups):
			# shared_labels: 0 means the attributes should not be shared, otherwise it should
			bool_share = shared_masks[group_id].squeeze()

			# print(bool_share)

			# swap the encodings that should be shared
			# # cyclic
			inputs0_swap[group_id, bool_share] = inputs1[group_id, bool_share]
			inputs0_a_swap[group_id, bool_share] = inputs0[group_id, bool_share]
			inputs1_swap[group_id, bool_share] = inputs0_a[group_id, bool_share]
			# inputs1_swap[group_id, bool_share] = inputs0_a[group_id, bool_share]

			# swap those encodings between the positive and augmented
			# inputs0_swap[group_id, ~bool_share] = inputs0_a[group_id, ~bool_share]
			# inputs0_a_swap[group_id, ~bool_share] = inputs0[group_id, ~bool_share]

			# # detach those that should not be shared
			inputs0_swap[group_id, ~bool_share.squeeze()].detach()
			inputs0_a_swap[group_id, ~bool_share.squeeze()].detach()
			inputs1_swap[group_id, ~bool_share.squeeze()].detach()

			# Dims: [N_Emb, Emb_dim]
			group_embeddings = self.embeddings[group_id]

			# Calculate distances
			distances0 = (torch.sum(inputs0[group_id] ** 2, dim=1, keepdim=True)
			             + torch.sum(group_embeddings.weight ** 2, dim=1)
			             - 2 * torch.matmul(inputs0[group_id], group_embeddings.weight.t()))
			distances0_a = (torch.sum(inputs0_a[group_id] ** 2, dim=1, keepdim=True)
			             + torch.sum(group_embeddings.weight ** 2, dim=1)
			             - 2 * torch.matmul(inputs0_a[group_id], group_embeddings.weight.t()))
			distances1 = (torch.sum(inputs1[group_id] ** 2, dim=1, keepdim=True)
			             + torch.sum(group_embeddings.weight ** 2, dim=1)
			             - 2 * torch.matmul(inputs1[group_id], group_embeddings.weight.t()))
			distances0_emb.append(distances0)
			distances0_a_emb.append(distances0_a)
			distances1_emb.append(distances1)

		distances0_emb = torch.stack(distances0_emb).permute(1, 0, 2)
		distances0_a_emb = torch.stack(distances0_a_emb).permute(1, 0, 2)
		distances1_emb = torch.stack(distances1_emb).permute(1, 0, 2)

		return inputs0_swap.permute(1, 0, 2), \
		       inputs0_a_swap.permute(1, 0, 2), \
		       inputs1_swap.permute(1, 0, 2), \
		       (distances0_emb, distances0_a_emb, distances1_emb)

	def _compute_triplet_loss(self, inputs0, inputs0_a, inputs1, shared_masks):

		# shared_masks: [G, B]
		# [B, G, L] --> [G, B, L]
		inputs0 = inputs0.permute(1, 0, 2).contiguous()
		inputs0_a = inputs0_a.permute(1, 0, 2).contiguous()
		inputs1 = inputs1.permute(1, 0, 2).contiguous()

		distances0_emb = []
		distances0_a_emb = []
		distances1_emb = []

		triplet_loss_pos_aug = 0.   # distance between augmented sample and sample
		triplet_loss_pos_neg = 0.   # distance between negative sample and sample
		triplet_loss_aug_neg = 0.   # distance between negative sample and sample
		for group_id in range(self.num_proto_groups):

				# get input of one group, Dims: [B, L]
				input0 = inputs0[group_id]
				input0_a = inputs0_a[group_id]
				input1 = inputs1[group_id]

				# shared_labels: 0 means the attributes should not be shared, otherwise it should
				bool_share = shared_masks[group_id].squeeze()
				# print(bool_share)

				triplet_loss_pos_aug += self._stopgrad_dist(input0[bool_share], input0_a[bool_share])

				triplet_loss_pos_neg += self._stopgrad_dist(input0[bool_share], input1[bool_share])

				triplet_loss_aug_neg += self._stopgrad_dist(input0_a[bool_share], input1[bool_share])

				# just for plotting
				# Dims: [N_Emb, Emb_dim]
				group_embeddings = self.embeddings[group_id]
				# Calculate distances
				distances0 = (torch.sum(input0 ** 2, dim=1, keepdim=True)
				             + torch.sum(group_embeddings.weight ** 2, dim=1)
				             - 2 * torch.matmul(input0, group_embeddings.weight.t()))
				distances0_a = (torch.sum(input0_a ** 2, dim=1, keepdim=True)
				             + torch.sum(group_embeddings.weight ** 2, dim=1)
				             - 2 * torch.matmul(input0_a, group_embeddings.weight.t()))
				distances1 = (torch.sum(input1 ** 2, dim=1, keepdim=True)
				             + torch.sum(group_embeddings.weight ** 2, dim=1)
				             - 2 * torch.matmul(input1, group_embeddings.weight.t()))
				distances0_emb.append(distances0)
				distances0_a_emb.append(distances0_a)
				distances1_emb.append(distances1)


		avg_triplet_loss = (triplet_loss_pos_aug + triplet_loss_pos_neg + triplet_loss_aug_neg)/ 3

		distances0_emb = torch.stack(distances0_emb).permute(1, 0, 2)
		distances0_a_emb = torch.stack(distances0_a_emb).permute(1, 0, 2)
		distances1_emb = torch.stack(distances1_emb).permute(1, 0, 2)

		return avg_triplet_loss, (distances0_emb, distances0_a_emb, distances1_emb)

	def _stopgrad_dist(self, x0, x1):
		d0 = torch.mean(
					(torch.sum(x0 ** 2, dim=1, keepdim=True)
			             + torch.sum(x1.detach() ** 2, dim=1)
			             - 2 * torch.matmul(x0, x1.t().detach())).diag()
				)
		d1 = torch.mean(
					(torch.sum(x0.detach() ** 2, dim=1, keepdim=True)
			             + torch.sum(x1 ** 2, dim=1)
			             - 2 * torch.matmul(x0.detach(), x1.t())).diag()
				)
		return (d0 + d1) / 2

	def _comp_shared_group_ids(self, inputs0, inputs0_a, inputs1, k=None):

		batch_size = inputs0.shape[0]

		# [B, G, C] --> [G, B, C]
		inputs0 = inputs0.permute(1, 0, 2).contiguous()
		inputs0_a = inputs0_a.permute(1, 0, 2).contiguous()
		inputs1 = inputs1.permute(1, 0, 2).contiguous()

		# compute the euclidian distance between the split encodings of each image from the pair
		distances_pos_neg = []
		distances_pos_aug = []
		distances_aug_neg = [] # just for plotting
		for group_id in range(self.num_proto_groups):

			# get input of one group, Dims: [B, H, W, C]
			input0 = inputs0[group_id]
			input0_a = inputs0_a[group_id]
			input1 = inputs1[group_id]

			# Calculate distances between encodings of each image
			distances_pos_neg.append((torch.sum(input0 ** 2, dim=1, keepdim=True)
			             + torch.sum(input1 ** 2, dim=1)
			             - 2 * torch.matmul(input0, input1.t())).diag())
			distances_pos_aug.append((torch.sum(input0 ** 2, dim=1, keepdim=True)
			             + torch.sum(input0_a ** 2, dim=1)
			             - 2 * torch.matmul(input0, input0_a.t())).diag())
			distances_aug_neg.append((torch.sum(input0_a ** 2, dim=1, keepdim=True)
			             + torch.sum(input1 ** 2, dim=1)
			             - 2 * torch.matmul(input0_a, input1.t())).diag())

		# [G, B]
		distances_pos_neg = torch.stack(distances_pos_neg)
		distances_pos_aug = torch.stack(distances_pos_aug)
		distances_aug_neg = torch.stack(distances_aug_neg)

		if k is None:
			# get mask of which group attributes to share between image encodings and which should be chosen individually
			# shared_masks = self._compute_shared_mask(distances_pos_neg)
			shared_masks = self._compute_shared_mask(distances_aug_neg)
		else:
			# or simply hard code that k groups should be shared
			# k = 1

			# for easier indexing: [G, B] --> [B, G]
			# distances = distances_pos_neg.permute(1, 0)
			distances = distances_aug_neg.permute(1, 0)

			# find k smallest distances per sample, specifying that the attributes of these groups should be shared between
			# the img pairs
			k_min_dist_group_ids = torch.topk(distances, k, largest=False, dim=1).indices

			# turn into index tensor where an index 1 means the attr should be shared, whereas 0 means the
			# group attribute should not be shared and thus chosen individually for each image
			# [B, G]
			shared_masks = torch.zeros((batch_size, self.num_proto_groups), dtype=torch.bool)
			for i in range(k):
				shared_masks[range(batch_size), k_min_dist_group_ids[:, i]] = True
			# now convert back: [B, G] --> [G, B]
			shared_masks = shared_masks.permute(1, 0)

		return shared_masks, (distances_pos_neg, distances_pos_aug, distances_aug_neg)

	def _compute_shared_mask(self, distances):
		# distances: [B, G]
		# compute threshold via heuristic similar to Locatello et al 2020
		thresholds = self._estimate_threshold(distances)
		# true if 'unchanged' and should be average
		shared_mask = distances < thresholds
		return shared_mask

	def _estimate_threshold(self, distances):
		"""
		Compute the threshold for each image pair in a batch of distances of all elements of the latent distributions.
		It should be noted that for a perfectly trained model, this threshold is always correct.
		distances: [B, G]
		"""
		# maximums = distances.max(axis=1, keepdim=True).values
		# minimums = distances.min(axis=1, keepdim=True).values
		maximums = distances.max(axis=0, keepdim=True).values
		minimums = distances.min(axis=0, keepdim=True).values
		return (0.5 * minimums) + (0.5 * maximums)


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
