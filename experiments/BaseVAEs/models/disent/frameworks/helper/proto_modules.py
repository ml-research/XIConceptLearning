import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from disent.frameworks.helper.autoencoder_helpers import list_of_distances


class VectorQuantizationLayer(nn.Module):
    def __init__(self, beta=0.25, input_dim=10, n_proto_vecs=(10,), device="cpu"):
        super(VectorQuantizationLayer, self).__init__()
        self.beta = beta
        self.n_proto_vecs = n_proto_vecs
        self.n_proto_groups = len(n_proto_vecs)
        self.device = device

        proto_vecs_list = []
        # print(n_proto_vecs)
        for num_protos in n_proto_vecs:
            p = torch.nn.Parameter(torch.rand(num_protos, input_dim, device=self.device))
            nn.init.xavier_uniform_(p, gain=1.0)
            proto_vecs_list.append(p)

        self.proto_vecs = nn.ParameterList(proto_vecs_list)

    def forward(self, latents):
        commitment_loss = torch.tensor(0., device=self.device)
        embedding_loss = torch.tensor(0., device=self.device)
        quantized_latents_recon = torch.zeros_like(latents, device=self.device) #[B, N, K]
        dists = dict()
        for k in range(len(self.proto_vecs)):
            # TODO: transforming dims from [B, N, K] to [N, B, K] could speed up indexing
            # compute the distance for each encoding slot to its corresponding group prototype codes
            dists[k] = list_of_distances(latents[:, k], self.proto_vecs[k])

            # Get the encoding that has the min distance
            encoding_inds = torch.argmin(dists[k], dim=1).unsqueeze(1)  # [BHW, 1]

            # Convert to one-hot encodings
            encoding_one_hot = torch.zeros(encoding_inds.size(0), self.n_proto_vecs[k], device=self.device)
            encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

            # Quantize the latents
            quantized_latents = torch.matmul(encoding_one_hot, self.proto_vecs[k])  # [BHW, D]

            # Compute the VQ Losses per group
            commitment_loss += F.mse_loss(quantized_latents.detach(), latents[:, k])
            embedding_loss += F.mse_loss(quantized_latents, latents[:, k].detach())

            # TODO: I don't understand why this is done
            # Add the residue back to the latents
            quantized_latents_recon[:, k] = latents[:, k] + (quantized_latents - latents[:, k]).detach()

        vq_loss = commitment_loss * self.beta + embedding_loss
        # TODO: should we do this? or just return summed loss?
        vq_loss /= self.n_proto_groups

        return quantized_latents_recon, vq_loss, dists


# TODO: experimental
class SoftminVectorQuantizationLayer(nn.Module):
    def __init__(self, beta=0.25, input_dim=10, n_proto_vecs=(10,), device="cpu"):
        super(SoftminVectorQuantizationLayer, self).__init__()
        self.beta = beta
        self.n_proto_vecs = n_proto_vecs
        self.n_proto_groups = len(n_proto_vecs)
        self.device = device

        self.softmin = torch.nn.Softmin(dim=1)
        self.softmin_temp = nn.Parameter(torch.tensor(1.0))
        self.softmin_temp.requires_grad = False

        proto_vecs_list = []
        # print(n_proto_vecs)
        for num_protos in n_proto_vecs:
            p = torch.nn.Parameter(torch.rand(num_protos, input_dim, device=self.device))
            nn.init.xavier_uniform_(p, gain=1.0)
            proto_vecs_list.append(p)

        self.proto_vecs = nn.ParameterList(proto_vecs_list)

    def forward(self, latents):
        commitment_loss = torch.tensor(0.)
        embedding_loss = torch.tensor(0.)
        quantized_latents = torch.zeros_like(latents) #[B, N, K]
        dists = dict()
        for k in range(len(self.proto_vecs)):
            # TODO: transforming dims from [B, N, K] to [N, B, K] could speed up indexing
            # compute the distance for each encoding slot to its corresponding group prototype codes
            dists[k] = list_of_distances(latents[:, k], self.proto_vecs[k])

            # Quantize the latents by softmin
            quantized_latents[:, k] = self.softmin(self.softmin_temp * dists[k]) @ self.proto_vecs[k]

            # Compute the VQ Losses per group
            commitment_loss += F.mse_loss(quantized_latents[:, k].detach(), latents[:, k])
            embedding_loss += F.mse_loss(quantized_latents[:, k], latents[:, k].detach())

            # TODO: I don't understand why this is done
            # Add the residue back to the latents
            quantized_latents[:, k] = latents[:, k] + (quantized_latents[:, k] - latents[:, k]).detach()

        vq_loss = commitment_loss * self.beta + embedding_loss
        # TODO: should we do this? or just return summed loss?
        vq_loss /= self.n_proto_groups

        return quantized_latents, vq_loss, dists


class PrototypeLayer(nn.Module):
    def __init__(self, input_dim=10, n_proto_vecs=(10,), device="cpu"):
        super(PrototypeLayer, self).__init__()
        self.n_proto_vecs = n_proto_vecs
        self.n_proto_groups = len(n_proto_vecs)
        self.device = device

        proto_vecs_list = []
        # print(n_proto_vecs)
        for num_protos in n_proto_vecs:
            p = torch.nn.Parameter(torch.rand(num_protos, input_dim, device=self.device))
            nn.init.xavier_uniform_(p, gain=1.0)
            proto_vecs_list.append(p)

        self.proto_vecs = nn.ParameterList(proto_vecs_list)

    def forward(self, x):
        out = dict()
        for k in range(len(self.proto_vecs)):
            out[k] = list_of_distances(x[k], self.proto_vecs[k])
        return out


class ProtoAggregateLayer(nn.Module):
    def __init__(self, n_protos, dim_protos, device="cpu", layer_type='sum', train_pw=False):
        super(ProtoAggregateLayer, self).__init__()
        self.device = device
        if layer_type == 'sum':
            self.net = _ProtoAggSumLayer(n_protos, dim_protos, train_pw)
        elif layer_type == 'linear':
            self.net = _ProtoAggDenseLayer(n_protos, dim_protos)
        elif layer_type == 'conv':
            self.net = _ProtoAggChannelConvLayer(n_protos, dim_protos)
        elif layer_type == 'simple_attention':
            self.net = _ProtoAggAttentionLayer(n_protos, dim_protos)
        elif layer_type == 'attention':
            self.net = _ProtoAggAttentionLayer(n_protos, dim_protos)
            #raise ValueError('Aggregation layer type not supported. Please email wolfgang.stammer@cs.tu-darmstadt.de')
        else:
            raise ValueError('Aggregation layer type not supported. Please email wolfgang.stammer@cs.tu-darmstadt.de')

        self.net.to(self.device)

    def forward(self, x):
        return self.net(x)


class _ProtoAggSumLayer(nn.Module):
    def __init__(self, n_protos, dim_protos, train_pw, device="cpu"):
        super(_ProtoAggSumLayer, self).__init__()
        self.device = device
        self.weights_protos = torch.nn.Parameter(torch.ones(n_protos,
                                                                dim_protos))
        # self.weights_protos.requires_grad = train_pw

    def forward(self, x):
        out = torch.mul(self.weights_protos, x)  # [batch, n_groups, dim_protos]
        # average over the groups, yielding a combined prototype per sample
        out = torch.mean(out, dim=1)  # [batch, dim_protos]
        return out


class _ProtoAggDenseLayer(nn.Module):
    def __init__(self, n_protos, dim_protos, device="cpu"):
        super(_ProtoAggDenseLayer, self).__init__()
        self.device = device
        self.layer = torch.nn.Linear(n_protos*dim_protos, n_protos*dim_protos, bias=False)

    def forward(self, x):
        out = torch.flatten(x, start_dim=1)
        # average over the groups, yielding a combined prototype per sample
        out = self.layer(out)  # [batch, dim_protos]
        return out


class _ProtoAggChannelConvLayer(nn.Module):
    def __init__(self, n_protos, dim_protos, device="cpu"):
        super(_ProtoAggChannelConvLayer, self).__init__()
        self.device = device
        self.layer = torch.nn.Conv1d(dim_protos, dim_protos, kernel_size=n_protos)

    def forward(self, x):
        # average over the groups, yielding a combined prototype per sample
        out = x.permute(0, 2, 1)
        out = self.layer(out)  # [batch, dim_protos]
        out = out.squeeze()
        return out


class _ProtoAggAttentionSimpleLayer(nn.Module):
    def __init__(self, n_protos, dim_protos, device="cpu"):
        super(_ProtoAggAttentionSimpleLayer, self).__init__()
        self.device = device
        self.n_protos = n_protos
        self.dim_protos = dim_protos
        self.qTransf = nn.ModuleList([torch.nn.Linear(dim_protos, dim_protos) for _ in range(n_protos)])
        self.linear_out = torch.nn.Linear(dim_protos, dim_protos)

    def forward(self, x):
        # 3 tansformations on input then attention
        # x: [batch, n_groups, dim_protos]
        q = torch.stack([self.qTransf[i](x[:, i]) for i in range(self.n_protos)], dim=1)
        att = q  # torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.n_protos)
        out = torch.mul(F.softmax(att, dim=1), x)
        out = torch.sum(out, dim=1)
        return out


class _ProtoAggAttentionLayer(nn.Module):
    def __init__(self, n_protos, dim_protos, device="cpu"):
        super(_ProtoAggAttentionLayer, self).__init__()
        self.device = device
        self.n_protos = n_protos
        self.dim_protos = dim_protos
        self.qTransf = nn.ModuleList([torch.nn.Linear(dim_protos, dim_protos) for _ in range(n_protos)])
        self.kTransf = nn.ModuleList([torch.nn.Linear(dim_protos, dim_protos) for _ in range(n_protos)])
        self.vTransf = nn.ModuleList([torch.nn.Linear(dim_protos, dim_protos) for _ in range(n_protos)])
        self.linear_out = torch.nn.Linear(dim_protos, dim_protos)


    def forward(self, x):
        bs = x.size(0)
        # 3 tansformations on input then attention
        # x: [batch, n_groups, dim_protos]
        q = torch.stack([self.qTransf[i](x[:,i]) for i in range(self.n_protos)], dim=1)
        k = torch.stack([self.kTransf[i](x[:, i]) for i in range(self.n_protos)], dim=1)
        v = torch.stack([self.vTransf[i](x[:, i]) for i in range(self.n_protos)], dim=1)

        q = q.view(bs, self.n_protos, 1, self.dim_protos)  # q torch.Size([bs, groups, 1, dim_protos])
        k = k.view(bs, self.n_protos, 1, self.dim_protos)
        v = v.view(bs, self.n_protos, 1, self.dim_protos)

        k = k.transpose(1, 2) # q torch.Size([bs, 1, groups, dim_protos])
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # torch.Size([bs, 1, groups, groups])
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim_protos)

        # torch.Size([bs, 1, groups, groups])
        scores = F.softmax(scores, dim=-1)
        # torch.Size([bs, 1, groups, dim_protos])
        output = torch.matmul(scores, v)

        # torch.Size([bs, groups, dim_protos])
        concat = output.transpose(1, 2).contiguous() \
            .view(bs, -1, self.dim_protos)
        # torch.Size([bs, dim_protos])
        out = torch.sum(concat, dim=1)
        out = self.linear_out(out)
        return out


class DenseLayerSoftmax(nn.Module):
    def __init__(self, input_dim=10, output_dim=10):
        super(DenseLayerSoftmax, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.linear(x)
        return out


#----------------------------------------------------------------------------------------------------------------------#
"""
Misc functions
"""

def get_cum_group_ids(n_proto_vecs):
    group_ids = list(np.cumsum(n_proto_vecs))
    group_ids.insert(0, 0)
    group_ranges = []
    for k in range(len(n_proto_vecs)):
        group_ranges.append([group_ids[k], group_ids[k + 1]])
    return group_ranges
