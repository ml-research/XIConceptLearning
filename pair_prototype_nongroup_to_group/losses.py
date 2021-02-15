import torch
from autoencoder_helpers import list_of_distances


def r1_loss(prototype_vectors, feature_vectors_z, dim_proto, config):
	# draws prototype close to training example
	r1_loss = 0
	for k in range(config['n_prototype_groups']):
		r1_loss += torch.mean(torch.min(
			list_of_distances(prototype_vectors[k],
			                  feature_vectors_z.view(-1, dim_proto)),
			dim=1)[0])

	return r1_loss


def r2_loss(prototype_vectors, feature_vectors_z, dim_proto, config):
	# draws encoding close to prototype
	r2_loss = 0
	for k in range(config['n_prototype_groups']):
		r2_loss += torch.mean(torch.min(
			list_of_distances(feature_vectors_z.view(-1, dim_proto),
			                  prototype_vectors[k]),
			dim=1)[0])
	return r2_loss


def ad_loss(proto_vecs):
	loss_ad = 0
	for k in range(len(proto_vecs)):
		loss_ad += torch.mean(torch.sqrt(torch.sum(proto_vecs[k].T ** 2, dim=1)), dim=0)
	return loss_ad


def pair_loss(s):
	return torch.mean(torch.abs(1 - torch.sum(torch.pow(s, 0.5), dim=1)), dim=0)
