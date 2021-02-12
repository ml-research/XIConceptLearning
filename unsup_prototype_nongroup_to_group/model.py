import torch
import torch.nn as nn
import modules as modules

class RAE(nn.Module):
    def __init__(self, input_dim=(1, 1, 28,28), n_z=10, filter_dim=32,
                 n_proto_vecs=(10,),
                 train_pw=False, softmin_temp=1, agg_type='sum',
                 device="cpu"):
        super(RAE, self).__init__()

        self.input_dim = input_dim
        self.n_z = n_z
        self.filter_dim = filter_dim
        self.n_proto_vecs = n_proto_vecs
        self.n_proto_groups = len(n_proto_vecs)
        self.device = device

        # encoder
        self.enc = modules.Encoder(input_dim=input_dim[1], filter_dim=filter_dim, output_dim=n_z)
        
        # forward encoder to determine input dim for prototype layer
        self.enc_out = self.enc.forward(torch.randn(input_dim))
        self.latent_shape = list(self.enc_out.shape[1:])
        self.dim_proto = self.enc_out.view(-1,1).shape[0]

        # prototype layer
        self.proto_layer = modules.PrototypeLayer(input_dim=self.dim_proto,
                                                  n_proto_vecs=self.n_proto_vecs,
                                                  device=self.device)

        self.proto_agg_layer = modules.ProtoAggregateLayer(n_protos=self.n_proto_groups,
                                                           dim_protos=self.dim_proto,
                                                           train_pw=train_pw,
                                                           layer_type=agg_type)

        # decoder
        # use forwarded encoder to determine output shapes for decoder
        dec_out_shapes = []
        for module in self.enc.modules():
            if isinstance(module, modules.ConvLayer):
                dec_out_shapes += [list(module.in_shape)]
        self.dec = modules.Decoder(input_dim=n_z, filter_dim=filter_dim,
                                   output_dim=input_dim[1], out_shapes=dec_out_shapes)

        self.softmin = torch.nn.Softmin(dim=1)
        self.softmin_temp = nn.Parameter(torch.ones(softmin_temp,))
        self.softmin_temp.requires_grad = False

    def comp_weighted_prototype_per_group(self, dists, proto_vecs):
        """
        Computes the softmin over the distances within a group and weights each prototype by this weighting.
        :param dists:
        :param prototype_vectors:
        :return:
        """
        # within every group compute the softmin over the distances and mutmul with the relevant prototype vectors
        # yielding a weighted prototype per group per training sample
        proto_vecs_softmin = torch.zeros(len(dists[0]),
                                         self.n_proto_groups,
                                         self.dim_proto,
                                         device=self.device)

        # stores the softmin weights
        s_weights = dict()
        for k in range(self.n_proto_groups):
            s_weights[k] = self.softmin(self.softmin_temp*dists[k])
            proto_vecs_softmin[:, k] = s_weights[k] @ proto_vecs[k]

        return proto_vecs_softmin, s_weights

    def comp_combined_prototype_per_sample(self, proto_vecs_softmin):
        """
        Given the softmin weighted prototypes per group, compute the mixture of these prototypes to combine to one
        prototype.
        :param prototype_vectors_softmin:
        :return:
        """
        out = self.proto_agg_layer(proto_vecs_softmin)
        return out

    def dec_prototypes(self, prototypes):
        """
        Wrapper helper for decoding the prototypes, helpful due to reshaping.
        :param prototypes:
        :return:
        """
        return self.dec(prototypes.reshape([prototypes.shape[0]] + self.latent_shape))

    def forward_decoder(self, latent_encoding, dists):
        # compute the softmin weighted prototype per group
        proto_vecs_softmin, s_weights = self.comp_weighted_prototype_per_group(dists,
                                                                               self.proto_layer.proto_vecs)  # [batch, n_group, dim_proto]

        # combine the prototypes per group to a single prototype, i.e. a mixture over all group prototypes
        agg_proto = self.comp_combined_prototype_per_sample(proto_vecs_softmin)  # [batch, dim_proto]

        # decode mixed prototypes
        recon_proto = self.dec_prototypes(agg_proto)

        # decode latent encoding
        recon_img = self.dec(latent_encoding)

        return recon_img, recon_proto, agg_proto, s_weights

    def forward_encoder(self, x):
        return self.enc(x)

    def forward(self, x, std=None):
        latent_enc = self.forward_encoder(x)
        # compute the distance of the training example to the prototype of each group
        dists = self.proto_layer(latent_enc.view(-1, self.dim_proto)) # [batch, n_group, n_proto]

        if std:
            # add gaussian noise to prototypes to avoid local optima
            for i in range(self.n_proto_groups):
                dists[i] += torch.normal(torch.zeros_like(dists[i]), std) # [batch, n_group, n_proto]

        recon_img, recon_proto, agg_proto, s_weights = self.forward_decoder(latent_enc, dists)

        res_dict = self.create_res_dict(self, recon_img, recon_proto, dists, s_weights, latent_enc,
                                        self.proto_layer.proto_vecs, agg_proto)

        return res_dict

    def create_res_dict(self, recon_img, recon_proto, dists, s_weights, latent_enc, proto_vecs, agg_proto):
        res_dict = {'recon_imgs': recon_img,
                    'recon_protos': recon_proto,
                    'dists': dists,
                    's_weights': s_weights,
                    'latent_enc': latent_enc,
                    'proto_vecs': proto_vecs,
                    'agg_protos': agg_proto}
        return res_dict
