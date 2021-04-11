import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.modules_proto as proto_modules
import models.modules as modules
from models.gproto_ae_sup import GProtoAE

class VectorQuantizerPair(nn.Module):
	def __init__(self, num_groups, num_embeddings, embedding_dim, commitment_cost, device):
		super(VectorQuantizerPair, self).__init__()

		self.num_groups = num_groups
		self._embedding_dim = embedding_dim
		self._num_embeddings = num_embeddings
		self.device = device

		self.embeddings = dict()
		for group_id in range(num_groups):
			self.embeddings[group_id] = nn.Embedding(self._num_embeddings, self._embedding_dim).to(self.device)
			# TODO: should we maybe initialize such that each prototype embedding is equidistantly far apart?
			self.embeddings[group_id].weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
		self._commitment_cost = commitment_cost

	# TODO: add std on to distances?
	def forward_pair(self, inputs0, inputs1, shared_labels=None):
		# shared_labels: [G, B]
		# [B, G, L] --> [G, B, L]
		inputs0 = inputs0.permute(1, 0, 2).contiguous()
		inputs1 = inputs1.permute(1, 0, 2).contiguous()

		vq_loss = 0.
		pair_loss = 0.
		perplexity = 0.
		quantized_all = []
		enc_embeddings_all = []
		enc_ids_one_hot_all = []
		distances0_all = []
		distances1_all = []
		for group_id in range(self.num_groups):

			# get input of one group, Dims: [B, L]
			input0 = inputs0[group_id]
			input1 = inputs1[group_id]

			label = shared_labels[group_id].unsqueeze(dim=1)
			label_bool_shared = label > -1
			label_bool_non_shared = label == -1

			# [B, L]
			input_shape = input0.shape

			# Dims: [N_Emb, Emb_dim]
			group_embeddings = self.embeddings[group_id]

			# Calculate distances
			distances0 = (torch.sum(input0 ** 2, dim=1, keepdim=True)
			             + torch.sum(group_embeddings.weight ** 2, dim=1)
			             - 2 * torch.matmul(input0, group_embeddings.weight.t()))
			distances1 = (torch.sum(input1 ** 2, dim=1, keepdim=True)
			             + torch.sum(group_embeddings.weight ** 2, dim=1)
			             - 2 * torch.matmul(input1, group_embeddings.weight.t()))

			distances0_all.append(distances0)
			distances1_all.append(distances1)

			encoding_indices0 = torch.zeros_like(label)
			encoding_indices1 = torch.zeros_like(label)

			# set to those that should be shared
			encoding_indices0[label_bool_shared] = label[label_bool_shared]
			encoding_indices1[label_bool_shared] = label[label_bool_shared]

			# get prototype indices for each image for non shared ids
			encoding_indices0[label_bool_non_shared] = torch.argmin(distances0, dim=1).unsqueeze(dim=1)[label_bool_non_shared]
			encoding_indices1[label_bool_non_shared] = torch.argmin(distances1, dim=1).unsqueeze(dim=1)[label_bool_non_shared]

			encodings0 = torch.zeros(encoding_indices0.shape[0], self._num_embeddings, device=inputs0.device)
			encodings0.scatter_(1, encoding_indices0, 1)
			encodings1 = torch.zeros(encoding_indices1.shape[0], self._num_embeddings, device=inputs1.device)
			encodings1.scatter_(1, encoding_indices1, 1)
			# store encoding ids
			enc_ids_one_hot_all.append(torch.stack([encodings0, encodings1]))

			# Quantize and unflatten
			# [B, L]
			quantized0 = torch.matmul(encodings0, group_embeddings.weight).view(input_shape)
			quantized1 = torch.matmul(encodings1, group_embeddings.weight).view(input_shape)
			enc_embeddings_all.append(torch.stack([quantized0, quantized1]))

			# VQ Loss (original)
			# e_latent_loss0 = F.mse_loss(quantized0.detach(), input0)
			# q_latent_loss0 = F.mse_loss(quantized0, input0.detach())
			# e_latent_loss1 = F.mse_loss(quantized1.detach(), input1)
			# q_latent_loss1 = F.mse_loss(quantized1, input1.detach())
			# vq_loss += 0.5 * (q_latent_loss0 + self._commitment_cost * e_latent_loss0 +
			#                q_latent_loss1 + self._commitment_cost * e_latent_loss1)
			# only update those encodings and prototypes that should be shared
			e_latent_loss0_shared = F.mse_loss(quantized0[label_bool_shared.squeeze()].detach(),
			                                   input0[label_bool_shared.squeeze()])
			q_latent_loss0_shared = F.mse_loss(quantized0[label_bool_shared.squeeze()],
			                                   input0[label_bool_shared.squeeze()].detach())
			e_latent_loss1_shared = F.mse_loss(quantized1[label_bool_shared.squeeze()].detach(),
			                                   input1[label_bool_shared.squeeze()])
			q_latent_loss1_shared = F.mse_loss(quantized1[label_bool_shared.squeeze()],
			                                   input1[label_bool_shared.squeeze()].detach())
			vq_loss += 0.5 * (q_latent_loss0_shared + self._commitment_cost * e_latent_loss0_shared +
			               q_latent_loss1_shared + self._commitment_cost * e_latent_loss1_shared)

			# # Pair Loss
			# # pair_loss += F.mse_loss(input0[label].detach(), input1[label])
			# # Pair Loss: enforce dissamilarity between encodings of non shared attributes
			# # TODO: also enforce similarity between shared attributes explicitly rather than implicitly via VQ loss?
			# pair_loss += torch.mean(F.cosine_similarity(input0[~label], input1[~label], dim=1))

			quantized0 = input0 + (quantized0 - input0).detach()
			quantized1 = input1 + (quantized1 - input1).detach()
			avg_probs0 = torch.mean(encodings0, dim=0)
			avg_probs1 = torch.mean(encodings1, dim=0)
			perplexity += 0.5 * (torch.exp(-torch.sum(avg_probs0 * torch.log(avg_probs0 + 1e-10))) +
			                     torch.exp(-torch.sum(avg_probs1 * torch.log(avg_probs1 + 1e-10))))

			quantized_all.append(torch.stack([quantized0.contiguous(), quantized1.contiguous()]))

		avg_vq_loss = vq_loss / self.num_groups
		avg_pair_loss = pair_loss / self.num_groups
		avg_perplexity = perplexity / self.num_groups
		# convert quantized from [G, P, B, L] -> [P, B, G, L]
		quantized_all = torch.stack(quantized_all).permute(1, 2, 0, 3)
		# [G, P, B, L] --> [P, G, B, L]
		enc_embeddings_all = torch.stack(enc_embeddings_all).permute(1, 0, 2, 3)
		enc_ids_one_hot_all = torch.stack(enc_ids_one_hot_all)
		distances0_all = torch.stack(distances0_all)
		distances1_all = torch.stack(distances1_all)

		print("\n---------------")
		print(distances0_all[:, :1].squeeze().detach().cpu().numpy())
		print(distances1_all[:, :1].squeeze().detach().cpu().numpy())
		print(enc_ids_one_hot_all[:, 0, :1].squeeze().detach().cpu().numpy())
		print(enc_ids_one_hot_all[:, 1, :1].squeeze().detach().cpu().numpy())
		print("---------------")

		return avg_vq_loss, avg_pair_loss, quantized_all, avg_perplexity, enc_embeddings_all, \
		       enc_ids_one_hot_all, (distances0_all, distances1_all)

	def forward_inference_single(self, inputs):
		# [B, G, L] --> [G, B, L]
		inputs = inputs.permute(1, 0, 2).contiguous()

		vq_loss = 0.
		perplexity = 0.
		quantized_all = []
		enc_embeddings_all = []
		enc_ids_one_hot_all = []
		distances_all = []
		for group_id in range(self.num_groups):
			# get input of one group, Dims: [B, L]
			input = inputs[group_id]

			# [B, L]
			input_shape = input.shape

			# Dims: [N_Emb, Emb_dim]
			group_embeddings = self.embeddings[group_id]

			# Calculate distances
			distances = (torch.sum(input ** 2, dim=1, keepdim=True)
			              + torch.sum(group_embeddings.weight ** 2, dim=1)
			              - 2 * torch.matmul(input, group_embeddings.weight.t()))

			distances_all.append(distances)

			# get prototype indices for each image for non shared ids
			encoding_indices = torch.argmin(distances, dim=1).unsqueeze(dim=1)

			encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
			encodings.scatter_(1, encoding_indices, 1)
			# store encoding ids
			enc_ids_one_hot_all.append(encodings)

			# Quantize and unflatten
			# [B, L]
			quantized = torch.matmul(encodings, group_embeddings.weight).view(input_shape)
			enc_embeddings_all.append(quantized)

			# VQ Loss (original)
			e_latent_loss = F.mse_loss(quantized.detach(), input)
			q_latent_loss = F.mse_loss(quantized, input.detach())
			vq_loss += q_latent_loss + self._commitment_cost * e_latent_loss

			quantized = input + (quantized - input).detach()
			avg_probs = torch.mean(encodings, dim=0)
			perplexity += torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

			quantized_all.append(quantized.contiguous())

		avg_vq_loss = vq_loss / self.num_groups
		avg_perplexity = perplexity / self.num_groups
		# [G, B, L] --> [B, G, L]
		quantized_all = torch.stack(quantized_all).permute(1, 0, 2)
		# [G, B, L]
		enc_embeddings_all = torch.stack(enc_embeddings_all)
		enc_ids_one_hot_all = torch.stack(enc_ids_one_hot_all)
		distances_all = torch.stack(distances_all)
		return avg_vq_loss, quantized_all, avg_perplexity, enc_embeddings_all, enc_ids_one_hot_all, distances_all


# TODO: add passing unsimilar number of prototypes per group
class GProtoAEPair(nn.Module):
	def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
	             num_groups, num_protos, commitment_cost, agg_type, device='cpu'):
		super(GProtoAEPair, self).__init__()

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

		self._decoder_proto = modules.Decoder(num_hiddens,
		                        num_hiddens,
		                        num_residual_layers,
		                        num_residual_hiddens)
		self._decoder_z = modules.Decoder(num_hiddens,
		                        num_hiddens,
		                        num_residual_layers,
		                        num_residual_hiddens)

		self.agg_layer = proto_modules.ProtoAggregateLayer(n_protos=self.num_groups, dim_protos=self.proto_dim,
		                                        train_pw=False,
		                                        layer_type=self.agg_type,
		                                        device=self.device)
		# self.attr_pred = nn.ModuleList([AttrPred(self.proto_dim, self.num_protos)
		#                                 for i in range(self.num_groups)])

		self._vq_vae = VectorQuantizerPair(self.num_groups, self.num_protos, self.proto_dim,
		                                   self.commitment_cost, device=self.device)

	def forward(self, imgs, shared_labels=None):
		if type(imgs) is not tuple:
			return self.forward_inference_single(imgs)
		else:
			return self.forward_pairs(imgs, shared_labels)

	def forward_pairs(self, imgs, shared_labels):
		(x0, x1) = imgs

		z0 = self._encoder(x0) #[B, F, W, H]
		z1 = self._encoder(x1) #[B, F, W, H]

		# TODO: add attention over feature space rather than linear?
		# [B, F, W, H] --> [B, G, D_P], D_P = F*W*H
		z0 = self.split(torch.flatten(z0, start_dim=1)).view(-1, self.num_groups, self.proto_dim)
		z1 = self.split(torch.flatten(z1, start_dim=1)).view(-1, self.num_groups, self.proto_dim)

		# reconstruct z directly
		# z_recon = self._decoder_z(z) # [B, 3, W, H]
		z_recon0 = self._decoder_z(self.agg_layer(z0).view(-1, self.latent_shape[0],
		                                                 self.latent_shape[1], self.latent_shape[2])) # [B, 3, W, H]
		z_recon1 = self._decoder_z(self.agg_layer(z1).view(-1, self.latent_shape[0],
		                                                 self.latent_shape[1], self.latent_shape[2])) # [B, 3, W, H]

		# shared_labels = self._comp_shared_group_ids(z0, z1)

		vq_loss, pair_loss, quantized, perplexity, embeddings, \
		embedding_ids_one_hot, distances = self._vq_vae.forward_pair(z0, z1, shared_labels)

		# aggregate the quantized tensors of several groups into one tensor:
		quantized0 = self.agg_layer(quantized[0]).view(-1, self.latent_shape[0],
		                                                 self.latent_shape[1], self.latent_shape[2]) # [B, G, D_P]
		quantized1 = self.agg_layer(quantized[1]).view(-1, self.latent_shape[0],
		                                                 self.latent_shape[1], self.latent_shape[2]) # [B, G, D_P]
		# reconstruct
		proto_recon0 = self._decoder_proto(quantized0) # [B, 3, W, H]
		proto_recon1 = self._decoder_proto(quantized1) # [B, 3, W, H]

		return vq_loss, pair_loss, (z_recon0, z_recon1), (proto_recon0, proto_recon1), perplexity, \
		       embeddings, embedding_ids_one_hot, distances

	def forward_inference_single(self, imgs):
		z = self._encoder(imgs) #[B, F, W, H]

		# TODO: add attention over feature space rather than linear?
		# [B, F, W, H] --> [B, G, D_P], D_P = F*W*H
		z = self.split(torch.flatten(z, start_dim=1)).view(-1, self.num_groups, self.proto_dim)

		# reconstruct z directly
		# z_recon = self._decoder_z(z) # [B, 3, W, H]
		z_recon = self._decoder_z(self.agg_layer(z).view(-1, self.latent_shape[0],
		                                                 self.latent_shape[1], self.latent_shape[2])) # [B, 3, W, H]

		vq_loss, quantized, perplexity, embeddings, \
		embedding_ids_one_hot, distances = self._vq_vae.forward_inference_single(z)

		# aggregate the quantized tensors of several groups into one tensor:
		quantized = self.agg_layer(quantized).view(-1, self.latent_shape[0],
		                                                 self.latent_shape[1], self.latent_shape[2]) # [B, G, D_P]
		# reconstruct
		proto_recon = self._decoder_proto(quantized) # [B, 3, W, H]

		return vq_loss, z_recon, proto_recon, perplexity, embeddings, embedding_ids_one_hot, distances

	def _comp_shared_group_ids(self, inputs0, inputs1):

		batch_size = inputs0.shape[0]

		# [B, G, C] --> [G, B, C]
		inputs0 = inputs0.permute(1, 0, 2).contiguous()
		inputs1 = inputs1.permute(1, 0, 2).contiguous()

		distances = []
		for group_id in range(self.num_groups):

			# get input of one group, Dims: [B, H, W, C]
			input0 = inputs0[group_id]
			input1 = inputs1[group_id]

			# Calculate distances between encodings of each image
			distances.append((torch.sum(input0 ** 2, dim=1, keepdim=True)
			             + torch.sum(input1 ** 2, dim=1)
			             - 2 * torch.matmul(input0, input1.t())).diag())

		# [G, B] --> [B, G]
		distances = torch.stack(distances).permute(1, 0)

		# get mask of which group attributes to share between image encodings and which should be chosen individually
		# shared_labels = self._compute_shared_mask(distances)

		# or simply hard code that k groups should be shared
		k = 1

		# find k smallest distances per sample, specifying that the attributes of these groups should be shared between
		# the img pairs
		k_min_dist_group_ids = torch.topk(distances, k, largest=False, dim=1).indices

		# turn into index tensor where an index 1 means the attr should be shared, whereas 0 means the
		# group attribute should not be shared and thus chosen individually for each image
		# [B, G]
		shared_labels = torch.zeros(batch_size, self.num_groups, dtype=torch.bool)
		for i in range(k):
			shared_labels[range(batch_size), k_min_dist_group_ids[:, i]] = True

		return shared_labels

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
		maximums = distances.max(axis=1, keepdim=True).values
		minimums = distances.min(axis=1, keepdim=True).values
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
	num_training_updates = 15000

	num_hiddens = 64
	num_residual_hiddens = 32
	num_residual_layers = 2

	num_protos = 4
	num_groups = 2

	commitment_cost = 0.25

	# decay = 0.99
	decay = 0.0

	learning_rate = 1e-3

	model = GProtoAEPair(num_hiddens=num_hiddens, num_residual_layers=num_residual_layers,
	                     num_residual_hiddens=num_residual_hiddens, num_groups=num_groups,
	                     num_protos=num_protos, commitment_cost=commitment_cost,
	                     agg_type='linear', device='cpu').to('cpu')

	data0 = torch.rand(15, 3, 64, 64)
	data1 = torch.rand(15, 3, 64, 64)
	vq_loss, data_recon, perplexity = model((data0, data1))
