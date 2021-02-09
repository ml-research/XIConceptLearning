import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim=1, filter_dim=32, output_dim=10):
        super(Encoder, self).__init__()
        
        model = []
        model += [ConvLayer(input_dim, input_dim*filter_dim)]

        for i in range(0, 2):
            model += [ConvLayer(input_dim*filter_dim, input_dim*filter_dim)]
            
        model += [ConvLayer(input_dim*filter_dim, output_dim)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


class ConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvLayer, self).__init__()
        
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1)
        self.activation = nn.LeakyReLU() # ReLU()

    def forward(self, x):
        # 'SAME' padding as in tensorflow
        # set padding =0 in Conv2d
        # out = F.pad(x, (0, 0, 1, 2))
        self.in_shape = x.shape[-2:]
        out = self.conv(x)
        # print(out.shape)
        out = self.activation(out)
        return out 

class Decoder(nn.Module):
    def __init__(self, input_dim=10, filter_dim=32, output_dim=1, out_shapes=[]):
        super(Decoder, self).__init__()

        model = []

        model += [ConvTransLayer(input_dim, input_dim*filter_dim, output_shape=out_shapes[-1])]

        model += [ConvTransLayer(input_dim*filter_dim, input_dim*filter_dim, output_shape=out_shapes[-2])]
        model += [ConvTransLayer(input_dim*filter_dim, input_dim*filter_dim, output_shape=out_shapes[-3])]

        model += [ConvTransLayer(input_dim*filter_dim, output_dim, output_shape=out_shapes[-4], activation=nn.Sigmoid)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out

class ConvTransLayer(nn.Module):
    def __init__(self, input_dim, output_dim, output_shape, activation=nn.LeakyReLU, padding='none'):
        super(ConvTransLayer, self).__init__()
        self.output_shape = output_shape
        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2, padding=0) # 2 2 0
        self.activation = activation()
        self.padding = padding

    def forward(self, x):
        out = self.conv(x)
        # print(f'{out.shape} -> {self.output_shape}')
        # refer to tensorflow_reverse_engineer.py to see reasoning
        if self.padding == 'same':
            if (out.shape[-2:][0] != self.output_shape[0]) & (out.shape[-2:][1] != self.output_shape[1]):
                diff_x = out.shape[-2:][0] - self.output_shape[0]
                diff_y =  out.shape[-2:][1] - self.output_shape[1]
                out = out[:,:,diff_x:,diff_y:]
            # print(out.shape)
        # print(out.shape)
        out = self.activation(out)
        return out

class PrototypeLayer(nn.Module):
    def __init__(self, input_dim=10, n_prototype_vectors=10):
        super(PrototypeLayer, self).__init__()

        self.prototype_vectors = nn.Parameter(torch.rand(n_prototype_vectors, input_dim), requires_grad=True) # .to('cuda:5')

    def forward(self, x):
        return list_of_distances(x, self.prototype_vectors)

def list_of_norms(x):
        return torch.sum(torch.pow(x, 2),dim=1)

def list_of_distances(x, y):
    xx = list_of_norms(x).view(-1, 1)
    yy = list_of_norms(y).view(1, -1)
    return xx + yy -2 * torch.matmul(x, torch.transpose(y, 0, 1))

class DenseLayerSoftmax(nn.Module):
    def __init__(self, input_dim=10, output_dim=10):
        super(DenseLayerSoftmax, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.linear(x)
        # out = self.softmax(out)
        return out
