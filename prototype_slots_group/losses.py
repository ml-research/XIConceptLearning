import torch
from autoencoder_helpers import list_of_distances


def r1_loss_nongroup(prototype_vectors, feature_vectors_z, dim_proto, config):
	# draws prototype close to training example
	r1_loss = torch.mean(torch.min(
			list_of_distances(prototype_vectors,
							  feature_vectors_z.view(-1, dim_proto)),
			dim=1)[0])
	return r1_loss


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


def pair_cos_loss(attr_probs, group_ranges):
	"""
	Computes the pair loss based on the cosine similarity between the attribute prediction probabilities
	(softmax outputs).
	:param attr_probs: list of dict, [2 x {n_groups}], i.e. for each single image of the pair the dictionary contains
	the predicted attribute probabilities of each group. E.g. attr_prob[0][0] contains the tensor of probabilities for
	the first images af each pair for the first attribute group.
	:return:
	"""
	cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
	loss_pair = 0

	# TODO: hard coded for now, s.t. the attributes in the second group should be the same, whereas those of first
	#  group should be orthogonal to another
	# for that group for which the attributes predictions should be the same over both img pairs
	loss_pair += 1. - cos(attr_probs[0][:, group_ranges[1][0]: group_ranges[1][1]],
	                      attr_probs[1][:, group_ranges[1][0]: group_ranges[1][1]])
	# for that group for which the attribute predicitons should be orthogonal over both img pairs
	# loss_pair += cos(attr_probs[0][:, group_ranges[0][0]: group_ranges[0][1]],
	#                  attr_probs[1][:, group_ranges[0][0]: group_ranges[0][1]])

	# return mean over samples
	return torch.mean(loss_pair, dim=0)