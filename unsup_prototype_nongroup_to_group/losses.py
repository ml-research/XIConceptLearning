import torch
from autoencoder_helpers import list_of_distances


def r1_loss(prototype_vectors, feature_vectors_z, dim_prototype, config):
	# draws prototype close to training example
	r1_loss = 0
	for k in range(config['n_prototype_groups']):
		r1_loss += torch.mean(torch.min(
			list_of_distances(prototype_vectors[k],
			                  feature_vectors_z.view(-1, dim_prototype)),
			dim=1)[0])

	return r1_loss


def r2_loss(prototype_vectors, feature_vectors_z, dim_prototype, config):
	# draws encoding close to prototype
	r2_loss = 0
	for k in range(config['n_prototype_groups']):
		r2_loss += torch.mean(torch.min(
			list_of_distances(feature_vectors_z.view(-1, dim_prototype),
			                  prototype_vectors[k]),
			dim=1)[0])
	return r2_loss


def ad_loss():
	return 0
