import torch
import torch.nn as nn
import modules as modules

class CAE(nn.Module):
    def __init__(self, input_dim=(1, 1, 28,28), n_z=10, filter_dim=32, n_prototype_vectors=10, n_classes=10):
        super(CAE, self).__init__()

        # encoder
        self.enc = modules.Encoder(input_dim=input_dim[1], filter_dim=filter_dim, output_dim=n_z)
        
        # prototype layer
        # forward encoder to determine input dim for prototype layer
        self.input_dim_prototype = self.enc.forward(torch.randn(input_dim)).view(-1,1).shape[0]
        self.prototype_layer = modules.PrototypeLayer(input_dim=self.input_dim_prototype,
                                              n_prototype_vectors=n_prototype_vectors)

        # decoder
        # use forwarded encoder to determine output shapes for decoder
        dec_out_shapes = []
        for module in self.enc.modules():
            if isinstance(module, modules.ConvLayer):
                dec_out_shapes += [list(module.in_shape)]
        self.dec = modules.Decoder(input_dim=n_z, filter_dim=filter_dim,
                                   output_dim=input_dim[1], out_shapes=dec_out_shapes)

        # output layer
        # final fully connected layer with softmax activation
        self.fc = modules.DenseLayerSoftmax(input_dim=n_prototype_vectors, output_dim=n_classes)

        self.feature_vectors = None
        self.softmin = torch.nn.Softmin(dim=1)

    def forward(self, x):
        out = self.enc(x)
        self.feature_vectors = out
        out = self.prototype_layer(out.view(-1,self.input_dim_prototype))
        out = self.fc(out)
        return out


class RAE(nn.Module):
    def __init__(self, input_dim=(1, 1, 28,28), n_z=10, filter_dim=32,
                 n_prototype_vectors=(10,),
                 train_pw=False, device="cpu"):
        super(RAE, self).__init__()

        self.input_dim = input_dim
        self.n_z = n_z
        self.filter_dim = filter_dim
        self.n_prototype_vectors = n_prototype_vectors
        self.n_prototype_groups = len(n_prototype_vectors)
        self.device = device

        # encoder
        self.enc = modules.Encoder(input_dim=input_dim[1], filter_dim=filter_dim, output_dim=n_z)
        
        # forward encoder to determine input dim for prototype layer
        self.enc_out = self.enc.forward(torch.randn(input_dim))
        self.latent_shape = list(self.enc_out.shape[1:])
        self.dim_prototype = self.enc_out.view(-1,1).shape[0]

        # prototype layer
        self.prototype_layer = modules.PrototypeLayer(input_dim=self.dim_prototype,
                                                      n_prototype_vectors=self.n_prototype_vectors,
                                                      device=self.device)

        self.weights_prototypes = torch.nn.Parameter(torch.ones(len(self.prototype_layer.prototype_vectors),
                                                                self.prototype_layer.prototype_vectors[0].shape[1]))
        self.weights_prototypes.requires_grad = train_pw

        # decoder
        # use forwarded encoder to determine output shapes for decoder
        dec_out_shapes = []
        for module in self.enc.modules():
            if isinstance(module, modules.ConvLayer):
                dec_out_shapes += [list(module.in_shape)]
        self.dec = modules.Decoder(input_dim=n_z, filter_dim=filter_dim,
                                   output_dim=input_dim[1], out_shapes=dec_out_shapes)

        self.softmin = torch.nn.Softmin(dim=1)

    def comp_weighted_prototype_per_group(self, dists, prototype_vectors):
        """
        Computes the softmin over the distances within a group and weights each prototype by this weighting.
        :param dists:
        :param prototype_vectors:
        :return:
        """
        # within every group compute the softmin over the distances and mutmul with the relevant prototype vectors
        # yielding a weighted prototype per group per training sample
        prototype_vectors_softmin = torch.zeros(len(dists[0]),
                                                self.n_prototype_groups,
                                                self.dim_prototype,
                                                device=self.device)
        for j in range(self.n_prototype_groups):
            prototype_vectors_softmin[:, j] = self.softmin(dists[j]) @ prototype_vectors[j]
        return prototype_vectors_softmin

    def comp_combined_prototype_per_sample(self, prototype_vectors_softmin):
        """
        Given the softmin weighted prototypes per group, compute the mixture of these prototypes to combine to one
        prototype.
        :param prototype_vectors_softmin:
        :return:
        """
        # TODO: use linear instead? or attention?
        # multiply each weighted group prototype with the group weights
        prototype_vectors_softmin = torch.mul(self.weights_prototypes, prototype_vectors_softmin) # [batch, n_groups, dim_protos]
        # average over the groups, yielding a combined prototype per sample
        prototype_vectors_softmin = torch.mean(prototype_vectors_softmin, dim=1) # [batch, dim_protos]
        return prototype_vectors_softmin

    def dec_prototypes(self, prototypes):
        """
        Wrapper helper for decoding the prototypes, helpful due to reshaping.
        :param prototypes:
        :return:
        """
        return self.dec(prototypes.reshape([prototypes.shape[0]] + self.latent_shape))

    def forward(self, x, std=None):
        latent_encoding = self.enc(x)
        # compute the distance of the training example to the prototype of each group
        dists = self.prototype_layer(latent_encoding.view(-1, self.dim_prototype)) # [batch, n_group, n_proto]

        if std:
            # add gaussian noise to prototypes to avoid local optima
            for i in range(self.n_prototype_groups):
                dists[i] += torch.normal(torch.zeros_like(dists[i]), std) # [batch, n_group, n_proto]

        # compute the softmin weighted prototype per group
        prototype_vectors_softmin = self.comp_weighted_prototype_per_group(dists,
                                                                           self.prototype_layer.prototype_vectors) # [batch, n_group, dim_proto]

        # combine the prototypes per group to a single prototype, i.e. a mixture over all group prototypes
        mixed_prototypes = self.comp_combined_prototype_per_sample(prototype_vectors_softmin) # [batch, dim_proto]

        # decode mixed prototypes
        recon_proto = self.dec_prototypes(mixed_prototypes)

        recon_img = self.dec(latent_encoding)

        return recon_img, recon_proto, dists, latent_encoding, self.prototype_layer.prototype_vectors, mixed_prototypes
