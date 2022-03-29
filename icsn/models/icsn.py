import torch
import torch.nn as nn
import numpy as np
from icsn.models.torch_truncnorm.TruncatedNormal import TruncatedNormal


class iCSN(nn.Module):
	def __init__(self, encoder, decoder, n_proto_vecs=[4, 2],
	             lin_enc_size=256, proto_dim=1, softmax_temp=1., extra_mlp_dim=4,
	             multiheads=False, train_protos=False, temp_scheduler_step=10,
	             temp_scheduler_rate=0.2, image_size=64, device='cpu'):
		super(iCSN, self).__init__()
		assert temp_scheduler_rate < 1 and temp_scheduler_rate > 0

		self.n_proto_vecs = n_proto_vecs
		self.n_groups = len(n_proto_vecs)
		self.n_attrs = sum(n_proto_vecs)
		self.proto_dim = proto_dim
		self.lin_enc_size = lin_enc_size
		self.softmax_temp = softmax_temp
		self.device = device
		self.multiheads = multiheads
		self.extra_mlp_dim = extra_mlp_dim
		self.train_protos = train_protos
		self.image_size = image_size
		self.batch_size = 0 # dummy
		self.temp_scheduler_step = temp_scheduler_step # at which steps (e.g. epochs or batch) to perform an update
		self.temp_scheduler_rate = temp_scheduler_rate
		self.encoder = encoder
		self.decoder = decoder

		# positions in one hot label vector that correspond to a class, e.g. indices 0 - 4 are for different colors
		self.attr_positions = list(np.cumsum(self.n_proto_vecs))
		self.attr_positions.insert(0, 0)

		if isinstance(image_size, int):
			self.latent_shape = tuple(self.encoder(torch.rand(1, 1, image_size)).shape[1:])
		elif isinstance(image_size, tuple):
			self.latent_shape = tuple(self.encoder(torch.rand(1, 3, image_size[0], image_size[1])).shape[1:])
		self.latent_flat = np.prod(self.latent_shape)

		self.encoder_linear = nn.Sequential(
			nn.Linear(self.latent_flat, self.lin_enc_size),
			nn.BatchNorm1d(self.lin_enc_size),
			nn.ReLU(),
		)
		self.decoder_linear = nn.Sequential(
			nn.Linear(self.n_attrs + self.extra_mlp_dim, self.latent_flat),
			nn.BatchNorm1d(self.latent_flat),
			nn.ReLU(),
		)

		if not self.multiheads:
			# LeakyRelu as in MarioNette
			self.split_mlp = nn.Sequential(
				nn.Linear(self.lin_enc_size, self.n_groups * self.proto_dim),
				nn.LeakyReLU(),
			)
		elif self.multiheads:
			# Alternative:
			# each group should have its own mlp head
			self.split_mlps = nn.ModuleList([nn.Sequential(
				nn.Linear(self.lin_enc_size, self.proto_dim),
				nn.LeakyReLU(),
			) for i in range(self.n_groups)])

		self.extra_mlp = nn.Sequential(
				nn.Linear(self.lin_enc_size, 128),
				nn.ReLU(),
				nn.Linear(128, self.extra_mlp_dim),
				nn.Sigmoid(),
			)

		self.proto_layer = self.init_prototypes()

		self.softmax = nn.Softmax(dim=1)
		# Separate n_groups channels into n_groups (equivalent with InstanceNorm)
		self.group_norm = nn.GroupNorm(self.n_groups, self.n_groups, eps=1e-05)

	def init_prototypes(self):
		self.proto_dict = nn.ModuleDict()
		for group_id, num_protos in enumerate(self.n_proto_vecs):
			self.proto_dict[str(group_id)] = nn.Embedding(num_protos, self.proto_dim).to(self.device)

			# truncated normal sampling
			tn_dist = TruncatedNormal(loc=0., scale=.5, a=-1., b=1.)
			self.proto_dict[str(group_id)].weight.data = tn_dist.rsample([num_protos, self.proto_dim]).to(self.device)

			# protottypes are not learnable
			self.proto_dict[str(group_id)].weight.requires_grad = self.train_protos

	def update_softmax_temp(self, epoch):
		# if current step (e.g. epoch) is at relevant count then decrease temperature by temp_scheduler_rate
		if epoch > 0:
			self.softmax_temp = self.softmax_temp - \
		                    ((epoch % self.temp_scheduler_step) == 0.) * \
		                    (self.softmax_temp * (1.-self.temp_scheduler_rate))

	# ------------------------------------------------------------------------------------------------ #
	# fcts for forward passing

	def forward(self, kwargs):
		return self.forward_single(kwargs)

	def forward_pairs(self, data_dict):
		imgs = data_dict['data']
		shared_masks = data_dict['shared_masks']
		# if current step (e.g. epoch) is at relevant count then decrease temperature by temp_scheduler_rate
		# check if imgs is tuple
		assert(len(imgs) == 2)
		(x0, x1) = imgs # sample, negative sample

		self.batch_size = x0.shape[0]

		# x: [B, 3, 64, 64] --> [B, F, W, H] --> [B, D]
		z0 = self.encode(x0)
		z1 = self.encode(x1)

		# extracts additional information important for reconstruction, but not relevant as a concept
		z0_extra = self.extra_mlp(z0)
		z1_extra = self.extra_mlp(z1)

		# [B, D] --> [B, G, D_P], D_P = D/G
		z0 = self.split(z0)
		z1 = self.split(z1)

		# compute distance to prototype embeddings and return softmin distance as label prediction
		(z0_proto, z1_proto), (pred0, pred1, pred0_swap, pred1_swap) = self._comp_proto_dists(z0, z1, shared_masks)

		pred0 = torch.cat((pred0, z0_extra), dim=1)
		pred1 = torch.cat((pred1, z1_extra), dim=1)
		pred0_swap = torch.cat((pred0_swap, z0_extra), dim=1)
		pred1_swap = torch.cat((pred1_swap, z1_extra), dim=1)

		# convert codes and extra encoding via decoder to reconstruction
		z0_proto_recon = self.proto_decode(pred0)
		z1_proto_recon = self.proto_decode(pred1)
		z0_proto_recon_swap = self.proto_decode(pred0_swap)
		z1_proto_recon_swap = self.proto_decode(pred1_swap)

		# remove the continuous variables from the final prediction
		if self.extra_mlp_dim != 0.:
			pred0 = pred0[:, :-self.extra_mlp_dim]
			pred1 = pred1[:, :-self.extra_mlp_dim]

		return (pred0, pred1), (z0_proto_recon, z1_proto_recon, z0_proto_recon_swap, z1_proto_recon_swap)

	def forward_single(self, data_dict):
		imgs = data_dict['data']
		self.batch_size = imgs.shape[0]

		# x: [B, 3, 64, 64] --> [B, F, W, H] --> [B, D]
		z = self.encode(imgs)

		z_extra = self.extra_mlp(z)

		# [B, D] --> [B, G, D_P], D_P = D/G
		z = self.split(z)

		z_proto, preds = self._comp_proto_dists_single(z)

		preds = torch.cat((preds, z_extra), dim=1)

		z_proto_recon = self.proto_decode(preds)

		# remove the continuous variables from the final prediction
		if self.extra_mlp_dim != 0.:
			preds = preds[:, :-self.extra_mlp_dim]

		return preds, z_proto_recon

	# ------------------------------------------------------------------------------------------------ #
	# helper fcts for forward pass

	def get_prototype_embeddings(self, group_id):
		proto_embeddings = self.proto_dict[str(group_id)].weight
		# proto_embeddings = self.proto_layer_norm[group_id](proto_embeddings.permute(1, 0)).permute(1, 0)
		return proto_embeddings

	def split(self, z):
		if self.multiheads:
			z = torch.stack([self.split_mlps[i].forward(z) for i in range(self.n_groups)]).permute(1, 0, 2)
		else:
			z = self.split_mlp(z).reshape(-1, self.n_groups, self.proto_dim)

		z = self.group_norm(z)

		return z

	def encode(self, x):
		z = self.encoder(x) #[B, F, W, H]
		# z = self.encoder_linear(z.view(-1, self.latent_flat))
		z = self.encoder_linear(z.reshape(z.shape[0], self.latent_flat))
		return z

	def proto_decode(self, preds):
		# reconstruct z
		z_proto = self.decoder_linear(preds)

		z_proto_recon = self.decoder(z_proto.view([-1] + list(self.latent_shape)))  # [B, 3, W, H]

		return z_proto_recon

	# ------------------------------------------------------------------------------------------------ #
	# fcts for getting proto-code

	def _comp_proto_dists(self, inputs0, inputs1, shared_masks):
		"""
		Computes the distance between each encoding to the prototype vectors and creates a latent prototype vector for
		each image.
		:param inputs0:
		:param inputs1:
		:param shared_masks:
		:return:
		"""

		# shared_masks: [G, B]
		# [B, G, L] --> [G, B, L]
		inputs0 = inputs0.permute(1, 0, 2).contiguous()
		inputs1 = inputs1.permute(1, 0, 2).contiguous()

		# [B, G] --> [G, B]
		shared_masks = shared_masks.permute(1, 0)

		z0_protos = torch.clone(inputs0)
		z1_protos = torch.clone(inputs1)

		distances0_emb = torch.tensor([], device=self.device)
		distances0_emb_swap = torch.tensor([], device=self.device)
		distances1_emb = torch.tensor([], device=self.device)
		distances1_emb_swap = torch.tensor([], device=self.device)

		for group_id in range(self.n_groups):

				# get input of one group, Dims: [B, L]
				input0 = inputs0[group_id]
				input1 = inputs1[group_id]

				# Dims: [N_Emb, Emb_dim]
				proto_embeddings = self.get_prototype_embeddings(group_id)

				distances0 = self.softmax_dot_product(input0, proto_embeddings)
				distances1 = self.softmax_dot_product(input1, proto_embeddings)
				# print(np.round(distances0[0].detach().cpu().numpy(), 2))

				# if the group id is
				if group_id < shared_masks.shape[0]:
					# shared_labels: 0 means the attributes should not be shared, otherwise it should
					bool_share = shared_masks[group_id].squeeze()

					# swap the distances for those attributes to be shared
					distances0_swap = torch.clone(distances0)
					distances1_swap = torch.clone(distances1)
					distances0_swap[bool_share] = distances1[bool_share]
					distances1_swap[bool_share] = distances0[bool_share]
				else:
					# swap the distances for those attributes to be shared
					distances0_swap = torch.clone(distances0)
					distances1_swap = torch.clone(distances1)

				distances0_emb = torch.cat((distances0_emb, distances0), dim=1)
				distances1_emb = torch.cat((distances1_emb, distances1), dim=1)
				distances0_emb_swap = torch.cat((distances0_emb_swap, distances0_swap), dim=1)
				distances1_emb_swap = torch.cat((distances1_emb_swap, distances1_swap), dim=1)

				# get prototype indices for each image
				encoding_indices0 = torch.argmax(distances0, dim=1).unsqueeze(dim=1)
				encoding_indices1 = torch.argmax(distances1, dim=1).unsqueeze(dim=1)

		#TODO: commented out for efficiency
		#        quantized0 = proto_embeddings[encoding_indices0].squeeze(dim=1)
		#        quantized1 = proto_embeddings[encoding_indices1].squeeze(dim=1)
		#
		# 		z0_protos[group_id, :, :] = quantized0
		# 		z1_protos[group_id, :] = quantized1
		#
		# z0_protos = z0_protos.permute(1, 0, 2)
		# z1_protos = z1_protos.permute(1, 0, 2)
		z0_protos = None
		z1_protos = None

		return (z0_protos, z1_protos), (distances0_emb, distances1_emb, distances0_emb_swap, distances1_emb_swap)

	def _comp_proto_dists_single(self, inputs):
		"""
		Computes the distance between each encoding to the prototype vectors and creates a latent prototype vector for
		each image.
		:param inputs0:
		:param inputs1:
		:param shared_masks:
		:return:
		"""

		# shared_masks: [G, B]
		# [B, G, L] --> [G, B, L]
		inputs = inputs.permute(1, 0, 2).contiguous()

		z_protos = torch.clone(inputs)

		distances_emb = torch.tensor([], device=self.device)

		for group_id in range(self.n_groups):
				# get input of one group, Dims: [B, L]
				input = inputs[group_id]

				# Dims: [N_Emb, Emb_dim]
				proto_embeddings = self.get_prototype_embeddings(group_id)

				distances = self.softmax_dot_product(input, proto_embeddings)

				distances_emb = torch.cat((distances_emb, distances), dim=1)

				# get prototype indices for each image
				encoding_indices = torch.argmax(distances, dim=1).unsqueeze(dim=1)

				quantized = proto_embeddings[encoding_indices].squeeze(dim=1)

				z_protos[group_id, :, :] = quantized

		z_protos = z_protos.permute(1, 0, 2)

		return z_protos, distances_emb

	def softmax_dot_product(self, enc, protos):
		"""
		softmax product as in MarioNette (Smirnov et al. 2021)
		:param enc:
		:param protos:
		:return:
		"""
		norm_factor = torch.sum(
			torch.cat(
				[torch.exp(torch.sum(enc * protos[i], dim=1) / np.sqrt(self.proto_dim)).unsqueeze(dim=1)
				 for i in range(protos.shape[0])
				 ],
				dim=1
			),
			dim=1
		)
		sim_scores = torch.cat(
			[(torch.exp(torch.sum(enc * protos[i], dim=1) / np.sqrt(self.proto_dim)) / norm_factor).unsqueeze(dim=1)
			 for i in range(protos.shape[0])],
			dim=1
		)

		# apply extra softmax to possibly enforce one-hot encoding
		sim_scores = self.softmax((1./self.softmax_temp) * sim_scores)

		return sim_scores
