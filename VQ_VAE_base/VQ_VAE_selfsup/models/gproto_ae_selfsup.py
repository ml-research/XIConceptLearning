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
class GProtoAETriplet(nn.Module):
	def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
	             num_groups, num_protos, commitment_cost, agg_type, device='cpu'):
		super(GProtoAETriplet, self).__init__()

		self.num_groups = num_groups
		self.num_protos = num_protos
		self.device = device
		self.agg_type = agg_type
		self.commitment_cost = commitment_cost

		self._encoder = modules.Encoder(3, num_hiddens,
		                        num_residual_layers,
		                        num_residual_hiddens)
		self.latent_shape = tuple(self._encoder(torch.rand(1, 3, 64, 64)).shape[1:])
		self.proto_dim = np.prod(self.latent_shape)

		self.split = nn.Linear(self.proto_dim, num_groups * self.proto_dim)

		# self._decoder_proto = modules.Decoder(num_hiddens,
		#                         num_hiddens,
		#                         num_residual_layers,
		#                         num_residual_hiddens)
		self._decoder_z = modules.Decoder(num_hiddens,
		                        num_hiddens,
		                        num_residual_layers,
		                        num_residual_hiddens)

		self.agg_layer = proto_modules.ProtoAggregateLayer(n_protos=self.num_groups, dim_protos=self.proto_dim,
		                                        train_pw=False,
		                                        layer_type=self.agg_type,
		                                        device=self.device)
		# MLP prediction heads as in https://arxiv.org/pdf/2011.10566.pdf, one per sample of the triplet and per
		# samples on per group, i.e. 3 x n_groups
		self.prediction_heads = [
			[nn.Sequential(
				nn.Linear(self.proto_dim, self.proto_dim),
				nn.ReLU()
			) for i in range(self.num_groups)]
			for j in range(3)
		]

	def forward_triplet_cont(self, imgs, k):
		"""
		forward pass for contuous disentanglement, i.e. no prototypes
		"""
		(x0, a_x0, x1) = imgs # sample, augmented sample, negative sample

		z0 = self._encoder(x0) #[B, F, W, H]
		a_z0 = self._encoder(a_x0) #[B, F, W, H]
		z1 = self._encoder(x1) #[B, F, W, H]

		# TODO: add attention over feature space rather than linear?
		# [B, F, W, H] --> [B, G, D_P], D_P = F*W*H
		z0 = self.split(torch.flatten(z0, start_dim=1)).view(-1, self.num_groups, self.proto_dim)
		a_z0 = self.split(torch.flatten(a_z0, start_dim=1)).view(-1, self.num_groups, self.proto_dim)
		z1 = self.split(torch.flatten(z1, start_dim=1)).view(-1, self.num_groups, self.proto_dim)

		# reconstruct z directly
		z_recon0 = self._decoder_z(self.agg_layer(z0).view(-1, self.latent_shape[0],
		                                                 self.latent_shape[1], self.latent_shape[2])) # [B, 3, W, H]
		z_recona0 = self._decoder_z(self.agg_layer(a_z0).view(-1, self.latent_shape[0],
		                                                 self.latent_shape[1], self.latent_shape[2])) # [B, 3, W, H]
		z_recon1 = self._decoder_z(self.agg_layer(z1).view(-1, self.latent_shape[0],
		                                                 self.latent_shape[1], self.latent_shape[2])) # [B, 3, W, H]

		# estimate which attributes are shared
		shared_masks = self._comp_shared_group_ids(z0, z1, k)

		# compute distance between all pairs of z
		triplet_loss, distances_tuple = self._compute_triplet_loss(z0, a_z0, z1, shared_masks)

		return (z_recon0, z_recona0, z_recon1), distances_tuple, triplet_loss

	def _compute_triplet_loss(self, inputs0, a_inputs0, inputs1, shared_masks):

		# shared_masks: [G, B]
		# [B, G, L] --> [G, B, L]
		inputs0 = inputs0.permute(1, 0, 2).contiguous()
		a_inputs0 = a_inputs0.permute(1, 0, 2).contiguous()
		inputs1 = inputs1.permute(1, 0, 2).contiguous()

		triplet_loss_aug = 0.   # distance between augmented sample and sample
		triplet_loss_neg = 0.   # distance between negative sample and sample
		distances_aug = []
		distances_neg = []
		for group_id in range(self.num_groups):

				# get input of one group, Dims: [B, L]
				input0 = inputs0[group_id]
				a_input0 = a_inputs0[group_id]
				input1 = inputs1[group_id]

				# compute distances for logging
				distances_aug.append(F.cosine_similarity(input0, a_input0))
				distances_neg.append(F.cosine_similarity(input0, input1))

				# pass encodings through prediction heads as in https://arxiv.org/pdf/2011.10566.pdf
				input0_p = self.prediction_heads[0][group_id].forward(input0)
				a_input0_p = self.prediction_heads[1][group_id].forward(a_input0)
				input1_p = self.prediction_heads[2][group_id].forward(input1)

				# shared_labels: 0 means the attributes should not be shared, otherwise it should
				bool_share = shared_masks[group_id]

				# as in https://arxiv.org/pdf/2011.10566.pdf
				triplet_loss_aug += 0.5 * torch.mean(-1. * F.cosine_similarity(input0_p, a_input0.detach()))
				triplet_loss_aug += 0.5 * torch.mean(-1. * F.cosine_similarity(a_input0_p, input0.detach()))

				# Hint: nan can occur if no encodings should be shared in this group
				# update those encodings and prototypes that should be shared via negative cosine
				triplet_loss_neg += 0.5 * catch_nan(torch.mean(-1. *
					F.cosine_similarity(
						input0_p[bool_share.squeeze()],
						input1[bool_share.squeeze()].detach()
					)
				))
				triplet_loss_neg += 0.5 * catch_nan(torch.mean(-1. *
					F.cosine_similarity(
						input0[bool_share.squeeze()].detach(),
						input1_p[bool_share.squeeze()]
					)
				))
				# update those encodings and prototypes that should not be shared via positive cosine
				triplet_loss_neg += 0.5 * catch_nan(torch.mean(
					F.cosine_similarity(
						input0_p[~bool_share.squeeze()],
						input1[~bool_share.squeeze()].detach()
					)
				))
				triplet_loss_neg += 0.5 * catch_nan(torch.mean(
					F.cosine_similarity(
						input0[~bool_share.squeeze()].detach(),
						input1_p[~bool_share.squeeze()]
					)
				))

		avg_triplet_loss = triplet_loss_aug  + triplet_loss_neg
		distances_aug = torch.stack(distances_aug)
		distances_neg = torch.stack(distances_neg)
		return avg_triplet_loss, (distances_aug, distances_neg)

	def _comp_shared_group_ids(self, inputs0, inputs1, k=None):

		batch_size = inputs0.shape[0]

		# [B, G, C] --> [G, B, C]
		inputs0 = inputs0.permute(1, 0, 2).contiguous()
		inputs1 = inputs1.permute(1, 0, 2).contiguous()

		# compute the euclidian distance between the split encodings of each image from the pair
		distances = []
		for group_id in range(self.num_groups):

			# get input of one group, Dims: [B, H, W, C]
			input0 = inputs0[group_id]
			input1 = inputs1[group_id]

			# Calculate distances between encodings of each image
			distances.append(-1 * F.cosine_similarity(input0, input1, dim=1))
			# euclidian distance
			# distances.append((torch.sum(input0 ** 2, dim=1, keepdim=True)
			#              + torch.sum(input1 ** 2, dim=1)
			#              - 2 * torch.matmul(input0, input1.t())).diag())

		# [G, B]
		distances = torch.stack(distances)

		if k is None:
			# get mask of which group attributes to share between image encodings and which should be chosen individually
			shared_masks = self._compute_shared_mask(distances)
		else:
			# or simply hard code that k groups should be shared
			# k = 1

			# for easier indexing: [G, B] --> [B, G]
			distances = distances.permute(1, 0)

			# find k smallest distances per sample, specifying that the attributes of these groups should be shared between
			# the img pairs
			k_min_dist_group_ids = torch.topk(distances, k, largest=False, dim=1).indices

			# turn into index tensor where an index 1 means the attr should be shared, whereas 0 means the
			# group attribute should not be shared and thus chosen individually for each image
			# [B, G]
			shared_masks = torch.zeros((batch_size, self.num_groups), dtype=torch.bool)
			for i in range(k):
				shared_masks[range(batch_size), k_min_dist_group_ids[:, i]] = True
			# now convert back: [B, G] --> [G, B]
			shared_masks = shared_masks.permute(1, 0)

		return shared_masks

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

	def dec_proto_by_selection(self, selection_ids):
		prototypes = []
		for ids in selection_ids:
			tmp = []
			for group_id in range(len(ids)):
				tmp.append(self._vq_vae.embeddings[group_id].weight[ids[group_id]])
			prototypes.append(torch.stack(tmp))
		prototypes = torch.stack(prototypes)

		prototypes = self.agg_layer(prototypes).view(-1, self.latent_shape[0],
		                                                 self.latent_shape[1], self.latent_shape[2]) # [B, G, D_P]
		# reconstruct
		proto_recon = self._decoder_proto(prototypes) # [B, 3, W, H]
		return proto_recon


if __name__ == '__main__':
	batch_size = 256

	num_hiddens = 64
	num_residual_hiddens = 32
	num_residual_layers = 2

	num_protos = 4
	num_groups = 2

	commitment_cost = 0.25

	# decay = 0.99
	decay = 0.0

	learning_rate = 1e-3

	model = GProtoAETriplet(num_hiddens=num_hiddens, num_residual_layers=num_residual_layers,
	                     num_residual_hiddens=num_residual_hiddens, num_groups=num_groups,
	                     num_protos=num_protos, commitment_cost=commitment_cost,
	                     agg_type='sum', device='cpu').to('cpu')

	data0 = torch.rand(15, 3, 64, 64)
	dataa0 = torch.rand(15, 3, 64, 64)
	data1 = torch.rand(15, 3, 64, 64)
	data_recon = model.forward_triplet_cont((data0, dataa0, data1), k=1)
