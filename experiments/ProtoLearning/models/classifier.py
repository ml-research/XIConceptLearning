import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.modules as modules

class Classifier(nn.Module):
	def __init__(self, n_proto_vecs=[4, 2],
	             hidden_dim=256, device='cpu'):
		super(Classifier, self).__init__()

		self.n_proto_vecs = n_proto_vecs
		self.n_groups = len(n_proto_vecs)
		self.train_group = 0
		self.n_attrs = sum(n_proto_vecs)
		self.device = device

		self.linears = nn.ModuleList([nn.Sequential(
				nn.Linear(self.n_attrs, size*128),
				nn.LeakyReLU(),
				nn.Linear(size*128, size),
				nn.Softmax(dim=1),
			) for size in self.n_proto_vecs])


	# ------------------------------------------------------------------------------------------------ #
	# fcts for forward passing

	def forward(self, preds):
		classification = []
		for i, group_id in enumerate(self.n_proto_vecs):
			classification.append(self.linears[i](preds))
		classification = torch.cat(classification, dim=1)
		return classification

	