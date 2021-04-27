import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


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
        self.weights_protos.requires_grad = train_pw

    def forward(self, x):
        out = torch.mul(self.weights_protos, x)  # [batch, n_groups, dim_protos]
        # average over the groups, yielding a combined prototype per sample
        out = torch.mean(out, dim=1)  # [batch, dim_protos]
        return out


class _ProtoAggDenseLayer(nn.Module):
    def __init__(self, n_protos, dim_protos, device="cpu"):
        super(_ProtoAggDenseLayer, self).__init__()
        self.device = device
        self.layer = torch.nn.Linear(n_protos*dim_protos, dim_protos, bias=False)

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
