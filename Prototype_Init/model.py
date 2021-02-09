import torch.nn as nn
from network import *

class CAE(nn.Module):
    def __init__(self, input_dim=(1, 1, 28,28), n_z=10, filter_dim=32, n_prototype_vectors=10, n_classes=10):
        super(CAE, self).__init__()

        # encoder
        self.enc = Encoder(input_dim=input_dim[1], filter_dim=filter_dim, output_dim=n_z)
        
        # prototype layer
        # forward encoder to determine input dim for prototype layer
        self.input_dim_prototype = self.enc.forward(torch.randn(input_dim)).view(-1,1).shape[0]
        self.prototype_layer = PrototypeLayer(input_dim=self.input_dim_prototype, n_prototype_vectors=n_prototype_vectors).to('cuda:0')

        # decoder
        # use forwarded encoder to determine output shapes for decoder
        self.dec_out_shapes = []
        for module in self.enc.modules():
            if isinstance(module, ConvLayer):
                self.dec_out_shapes += [list(module.in_shape)]
        self.dec = Decoder(input_dim=n_z, filter_dim=filter_dim, output_dim=input_dim[1], out_shapes=self.dec_out_shapes)

        # output layer
        # final fully connected layer with softmax activation
        self.fc = DenseLayerSoftmax(input_dim=n_prototype_vectors, output_dim=n_classes)

        self.feature_vectors = None

    def forward(self, x):
        out = self.enc(x)
        self.feature_vectors = out
        out = self.prototype_layer(out.view(-1,self.input_dim_prototype))
        out = self.fc(out)
        return out

    def forward_dec(self, x):
        out = self.dec(x)
        return out
