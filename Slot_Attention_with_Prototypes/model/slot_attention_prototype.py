import numpy as np
import torch.nn as nn
from model.model_blocks import *
from model.slot_attention_obj_discovery import SlotAttention_model, SlotAttention_decoder #SlotAttention_model, SoftPositionEmbed, SlotAttention_decoder

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
        self.fc = DenseLayer(input_dim=n_prototype_vectors, output_dim=n_classes)

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

class SlotCAE(nn.Module):
    def __init__(self, input_dim=(1, 1, 28,28), filter_dim=32, n_prototype_vectors=10, n_classes=10, n_slots=2, n_iters=3, head_config=None):
        super(SlotCAE, self).__init__()

        # encoder
        self.n_slots = n_slots
        self.slot_att = SlotAttention_model(n_slots=11, n_iters=3, n_attr=18)
        slot_dim = self.slot_att.encoder_hidden_channels
        self.head_input_dim = slot_dim // len(head_config)

        class_heads = []
        for head in head_config:
            class_heads += [ClassHead(input_dim=self.head_input_dim, hidden_dim=slot_dim, n_classes=head['len']).to('cuda:0')]
        self.class_heads = nn.ModuleList(class_heads)

        # prototype layer
        # forward slot_attention_model to determine input dim for prototype layer
        # self.input_dim_prototype = self.slot_att.forward_enc(torch.randn(input_dim)).shape[-1]
        # self.prototype_layer = PrototypeLayer(input_dim=self.input_dim_prototype, n_prototype_vectors=n_prototype_vectors).to('cuda:0')

        # # output layer
        # # final fully connected layer with softmax activation
        # self.fc = DenseLayer(input_dim=n_prototype_vectors, output_dim=n_classes, activation=nn.Sigmoid)

        self.feature_vectors = None

    def get_feature_vectors(self, x):
        return self.feature_vectors

    def forward(self, x):
        out = self.slot_att(x, 'enc')
        # print('slot_out_shape', out.shape)
        self.feature_vectors = out

        heads_out = []
        for i, head in enumerate(self.class_heads):
            shape_old = out.shape
            in_prot = out.reshape(-1, shape_old[-1])
            # print('in_prot', in_prot.shape)
            out_prot = head(in_prot[:,i*self.head_input_dim:(i+1)*self.head_input_dim])
            # print('out_proto', out_prot.shape)
            out_tmp = out_prot.reshape(shape_old[0], shape_old[1], -1)
            # print('out', out_tmp.shape)
            heads_out += [out_tmp]
        # print(len(heads_out))
        
        return heads_out

    def forward_enc(self, x):
        out = self.slot_att(x, 'enc')
        return out

    def forward_dec(self, slots):
        return self.slot_att(slots, 'dec')

    def load_pretrained_slot_attention(self, log_path):
        log = torch.load(log_path)
        weights = self.convert_dataparallel_weights(log["weights"])
        self.slot_att.load_state_dict(weights, strict=True)

    def freeze_slot_attention(self):
        for child in self.slot_att.children():
            # if not isinstance(child, SlotAttention_decoder):
            for param in child.parameters():
                param.requires_grad = False
    	
    def convert_dataparallel_weights(self, weights):
        converted_weights = {}
        keys = weights.keys()
        for key in keys:
            new_key = key.split("module.")[-1]
            converted_weights[new_key] = weights[key]
        return converted_weights
