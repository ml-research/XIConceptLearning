import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoder_helpers import list_of_distances
import math
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_dim=1, filter_dim=32, output_dim=10):
        super(Encoder, self).__init__()

        model = []
        model += [ConvLayer(input_dim, filter_dim)]

        for i in range(0, 2):
            model += [ConvLayer(filter_dim, filter_dim)]

        model += [ConvLayer(filter_dim, output_dim, last_layer=True)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


class ConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, last_layer=False):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1)
        self.activation = nn.ReLU()
        self.last_layer = last_layer

    def forward(self, x):
        # 'SAME' padding as in tensorflow
        # set padding =0 in Conv2d
        # out = F.pad(x, (0, 0, 1, 2))
        self.in_shape = x.shape[-2:]
        out = self.conv(x)
        # print(x.shape)
        if not self.last_layer:
            out = self.activation(out)
        return out


class Decoder(nn.Module):
    def __init__(self, input_dim=10, filter_dim=32, output_dim=1, out_shapes=[]):
        super(Decoder, self).__init__()

        model = []

        model += [ConvTransLayer(input_dim, filter_dim, output_shape=out_shapes[-1], padding=(0,0,0,0))] # (0,1,0,1) ... tf-> (0,1,0,1)

        model += [ConvTransLayer(filter_dim, filter_dim, output_shape=out_shapes[-2], padding=(0,0,0,0))] # (0,1,0,1)
        model += [ConvTransLayer(filter_dim, filter_dim, output_shape=out_shapes[-3], padding=(0,0,0,0))] # (0,0,0,0)

        model += [ConvTransLayer(filter_dim, output_dim, output_shape=out_shapes[-4], padding=(0,0,0,0), activation=nn.Sigmoid)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


class ConvTransLayer(nn.Module):
    def __init__(self, input_dim, output_dim, output_shape, padding=(0,0,0,0), activation=nn.Sigmoid):
        super(ConvTransLayer, self).__init__()
        self.padding = padding
        self.output_shape = output_shape
        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=3, stride=2, padding=0) # 2 2 0
        self.activation = activation()

    def forward(self, x):
        out = F.pad(x, self.padding)
        out = self.conv(out)
        # print(f'{out.shape} -> {self.output_shape}')
        # refer to tensorflow_reverse_engineer.py to see reasoning
        if (out.shape[-2:][0] != self.output_shape[0]) & (out.shape[-2:][1] != self.output_shape[1]):
            diff_x = out.shape[-2:][0] - self.output_shape[0]
            diff_y =  out.shape[-2:][1] - self.output_shape[1]
            out = out[:,:,diff_x:,diff_y:]
        out = self.activation(out)
        return out


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
        # out = dict()
        # for k in range(len(self.proto_vecs)):
        #     out[k] = list_of_distances(x[k], self.proto_vecs[k])
        # return out
        pass

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


class DenseLayerSoftmax(nn.Module):
    def __init__(self, input_dim=10, output_dim=10):
        super(DenseLayerSoftmax, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.linear(x)
        return out


class AttributePredictor(nn.Module):
    def __init__(self, in_dim=40, n_proto_vecs=(4,), temp=0.01, device='cpu'):
        super(AttributePredictor, self).__init__()

        self.in_dim = in_dim
        self.out_dim = sum(n_proto_vecs)
        self.n_proto_vecs = n_proto_vecs
        self.n_proto_groups = len(n_proto_vecs)
        self.group_ranges = get_cum_group_ids(n_proto_vecs)
        self.temp = temp

        self.attr_net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(in_dim, in_dim),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(in_dim, self.out_dim)
        )

    def apply_softmax_per_attr(self, x):
        for ids in self.group_ranges:
            # x[:, ids[0]:ids[1]] = F.softmax(x[:, ids[0]:ids[1]]/self.temp, dim=1)
            # TODO: possibly set hard to False?
            x[:, ids[0]:ids[1]] = F.gumbel_softmax(x[:, ids[0]:ids[1]], tau=self.temp, dim=1, hard=False)
        return x

    def forward(self, x):
        # flatten image input
        x = x.view(-1, np.prod(list(x.shape[1:])))
        # forward through encoding
        x = self.attr_net(x)
        # x = F.softmax(x/self.temp, dim=1)
        x = self.apply_softmax_per_attr(x)
        return x


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
