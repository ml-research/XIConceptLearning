import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.modules_proto as proto_modules
import models.modules as modules

class VectorQuantizer(nn.Module):
	def __init__(self, num_groups, num_embeddings, embedding_dim, commitment_cost, device):
		super(VectorQuantizer, self).__init__()

		self.num_groups = num_groups
		self._embedding_dim = embedding_dim
		self._num_embeddings = num_embeddings
		self.device = device

		self.embeddings = dict()
		for group_id in range(num_groups):
			self.embeddings[group_id] = nn.Embedding(self._num_embeddings, self._embedding_dim).to(self.device)
			self.embeddings[group_id].weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
		self._commitment_cost = commitment_cost

	def forward(self, inputs, labels=None):
		# [B, G, L] --> [G, B, L]
		inputs = inputs.permute(1, 0, 2).contiguous()

		loss = 0.
		perplexity = 0.
		quantized_all = []
		enc_embeddings_all = []
		enc_ids_one_hot_all = []
		for group_id in range(self.num_groups):

			# get input of one group, Dims: [B, L]
			input = inputs[group_id]

			input_shape = input.shape

			# Dims: [N_P, D_P]
			group_embeddings = self.embeddings[group_id]

			# Calculate distances
			distances = (torch.sum(input ** 2, dim=1, keepdim=True)
			             + torch.sum(group_embeddings.weight ** 2, dim=1)
			             - 2 * torch.matmul(input, group_embeddings.weight.t()))

			if labels != None:
				encoding_indices = labels[group_id].unsqueeze(dim=1)
			else:
				# Encoding
				encoding_indices = torch.argmin(distances, dim=1).unsqueeze(dim=1)
			encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
			encodings.scatter_(1, encoding_indices, 1)
			# store encoding ids
			enc_ids_one_hot_all.append(encodings)

			# Quantize and unflatten
			quantized = torch.matmul(encodings, group_embeddings.weight).view(input_shape)
			enc_embeddings_all.append(quantized)

			# Loss
			e_latent_loss = F.mse_loss(quantized.detach(), input)
			q_latent_loss = F.mse_loss(quantized, input.detach())
			loss += q_latent_loss + self._commitment_cost * e_latent_loss

			quantized = input + (quantized - input).detach()
			avg_probs = torch.mean(encodings, dim=0)
			perplexity += torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

			quantized_all.append(quantized.contiguous())

		avg_loss = loss / self.num_groups
		avg_perplexity = perplexity / self.num_groups
		# convert quantized from [G, B, L] -> [B, G, L]
		quantized_all = torch.stack(quantized_all).permute(1, 0, 2)
		# [G, B, L]
		enc_embeddings_all = torch.stack(enc_embeddings_all)
		enc_ids_one_hot_all = torch.stack(enc_ids_one_hot_all)
		return avg_loss, quantized_all, avg_perplexity, enc_embeddings_all, enc_ids_one_hot_all


class GProtoAE(nn.Module):
	def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
	             num_groups, num_protos, commitment_cost, agg_type, device='cpu'):
		super(GProtoAE, self).__init__()

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

		# self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
		#                               out_channels=num_groups * embedding_dim,
		#                               kernel_size=1,
		#                               stride=1)
		self.split = nn.Linear(self.proto_dim, num_groups * self.proto_dim)
		self.unsplit = nn.Linear(num_groups * self.proto_dim, self.proto_dim)
		self._vq_vae = VectorQuantizer(num_groups, num_protos, self.proto_dim,
		                               commitment_cost, device=self.device)
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
		self.attr_pred = nn.ModuleList([AttrPred(self.proto_dim, self.num_protos)
		                                for i in range(self.num_groups)])

	def forward(self, x, labels=None):
		z = self._encoder(x) #[B, F, W, H]

		# TODO: add attention over feature space rather than cond2d?
		# # [B, F, W, H] --> [B, G, D_P], D_P = F*W*H
		# z = self._pre_vq_conv(z).view(-1, self.num_groups, self.embedding_dim, z.shape[-2], z.shape[-1])
		z = self.split(torch.flatten(z, start_dim=1)).view(-1, self.num_groups, self.proto_dim)

		# reconstruct z directly
		# z_recon = self._decoder_z(z) # [B, 3, W, H]
		z_recon = self._decoder_z(self.agg_layer(z).view(-1, self.latent_shape[0],
		                                                 self.latent_shape[1], self.latent_shape[2])) # [B, 3, W, H]

		loss, quantized, perplexity, embeddings, embedding_ids_one_hot = self._vq_vae(z, labels)
		# aggregate the quantized tensors of several groups into one tensor:
		quantized = self.agg_layer(quantized).view(-1, self.latent_shape[0],
		                                                 self.latent_shape[1], self.latent_shape[2]) # [B, G, D_P]
		# quantized = self.unsplit(torch.flatten(quantized, start_dim=1)).view(-1, self.latent_shape[0],
		#                                                                       self.latent_shape[1],
		#                                                                       self.latent_shape[2]) # [B F, W, H]
		# reconstruct
		proto_recon = self._decoder_proto(quantized) # [B, 3, W, H]

		# train to clasify the chosen prototype
		attr_preds = torch.stack([self.attr_pred[i].forward(embeddings[i])
		                          for i in range(self.num_groups)]) # [G, B, Num_Emb]

		return loss, z_recon, proto_recon, perplexity, embeddings, embedding_ids_one_hot, attr_preds

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

	embedding_dim = 32
	num_embeddings = 4
	num_groups = 2

	commitment_cost = 0.25

	# decay = 0.99
	decay = 0.0

	learning_rate = 1e-3

	model = GProtoVAE(num_hiddens=num_hiddens, num_residual_layers=num_residual_layers,
	                  num_residual_hiddens=num_residual_hiddens, num_groups=num_groups,
	                  num_protos=num_embeddings, commitment_cost=commitment_cost,
	                  agg_type='sum', device='cpu').to('cpu')

	data = torch.rand(15, 3, 64, 64)
	vq_loss, data_recon, perplexity = model(data)

	recon_error = F.mse_loss(data_recon, data)
	loss = recon_error + vq_loss
