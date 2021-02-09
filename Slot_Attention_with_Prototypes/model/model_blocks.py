import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.autoencoder_helpers import *

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

class SlotDecoder(nn.Module):
    def __init__(self, input_dim=10, filter_dim=32, output_dim=1, out_shapes=[]):
        super(SlotDecoder, self).__init__()

        model = []

        model += [ConvTransLayer(input_dim, input_dim*filter_dim)]

        model += [ConvTransLayer(input_dim*filter_dim, input_dim*filter_dim)]
        model += [ConvTransLayer(input_dim*filter_dim, input_dim*filter_dim)]

        model += [ConvTransLayer(input_dim*filter_dim, output_dim, activation=nn.Sigmoid)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out

class ConvTransLayer(nn.Module):
    def __init__(self, input_dim, output_dim, output_shape=None, activation=nn.LeakyReLU, padding='none'):
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
        out = self.activation(out)
        return out

class PrototypeLayer(nn.Module):
    def __init__(self, input_dim=10, n_prototype_vectors=10):
        super(PrototypeLayer, self).__init__()

        self.prototype_vectors = nn.Parameter(torch.rand(n_prototype_vectors, input_dim), requires_grad=True) # .to('cuda:5')

    def forward(self, x):
        return list_of_distances(x, self.prototype_vectors)

class DenseLayer(nn.Module):
    def __init__(self, input_dim=10, output_dim=10, activation=nn.Softmax, dim=1):
        super(DenseLayer, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        if activation == nn.Softmax:
            self.activation = activation(dim=dim)
        else:
            self.activation = activation()

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out

class ClassHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, n_prototype_vectors=0, activation=nn.Softmax):
        super(ClassHead, self).__init__()

        if n_prototype_vectors == 0:
            n_prototype_vectors = n_classes+1
        
        # square matrix
        # force square
        # hidden_dim = input_dim
        self.W = torch.nn.Parameter(torch.rand(input_dim, hidden_dim), requires_grad=True)
        # self.W = torch.nn.Parameter(torch.eye(input_dim), requires_grad=False)
        self.prototype_layer = PrototypeLayer(input_dim=hidden_dim, n_prototype_vectors=n_prototype_vectors)

        self.fc = DenseLayer(input_dim=n_prototype_vectors, output_dim=n_classes, activation=activation)

    def forward(self, x, op=None):

        if op == 'lin_trans':
            return self.forward_linear_transform(x)
        elif op == 'lin_trans_inv':
            return self.inverse_linear_transform(x)
        elif op =='prototype_vectors':
            return self.prototype_layer.prototype_vectors

        # print('head', x.shape)
        out = self.forward_linear_transform(x)
        # print('head trans', out.shape)
        out = self.prototype_layer(out)
        # print('head proto', out.shape)
        out = self.fc(out)
        return out

    def forward_linear_transform(self, x):
        return x @ self.W

    def inverse_linear_transform(self, x):
        try:
            return x # x @ torch.inverse(self.W)
        except:
            print('inverse failed, trying pseudo inverse')
            return x @ torch.inverse(self.W)
            try:
                return x @ torch.inverse(self.W)
            except:
                print('pseudo inverse failed, exiting')
                exit(42)