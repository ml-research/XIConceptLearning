import torch
import torch.nn as nn
import modules as modules
import autoencoder_helpers as ae_helpers

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

    def comp_min_prototype_per_group(self, dists, proto_vecs):
        """
        Computes the min over the distances within a group and weights each prototype by this weighting.
        :param dists:
        :param prototype_vectors:
        :return:
        """
        # within every group compute the softmin over the distances and mutmul with the relevant prototype vectors
        # yielding a weighted prototype per group per training sample
        # [batch, n_groups, dim_proto]
        proto_vecs_min = torch.zeros(len(dists[0]),
                                         self.n_proto_groups,
                                         self.dim_proto,
                                         device=self.device)

        # stores the softmin weights
        min_weights = dict()
        for k in range(self.n_proto_groups):
            min_weights[k] = torch.argmin(dists[k], dim=1)
            proto_vecs_min[:, k] = proto_vecs[k][min_weights[k]]

        return proto_vecs_min, min_weights

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
        proto_vecs_min, min_weights = self.comp_min_prototype_per_group(dists,
                                                                               self.proto_layer.proto_vecs)  # [batch, n_group, dim_proto]

        # combine the prototypes per group to a single prototype, i.e. a mixture over all group prototypes
        agg_proto = self.comp_combined_prototype_per_sample(proto_vecs_min)  # [batch, dim_proto]

        # decode mixed prototypes
        recon_proto = self.dec_prototypes(agg_proto)

        # decode latent encoding
        recon_img = self.dec(latent_encoding)

        return recon_img, recon_proto, agg_proto, min_weights

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

        recon_img, recon_proto, agg_proto, min_weights = self.forward_decoder(latent_enc, dists)

        res_dict = self.create_res_dict(recon_img, recon_proto, dists, min_weights, latent_enc,
                                        self.proto_layer.proto_vecs, agg_proto)

        return res_dict

    # TODO: rename s_weights to min_weights
    def create_res_dict(self, recon_img, recon_proto, dists, s_weights, latent_enc, proto_vecs, agg_proto):
        res_dict = {'recon_imgs': recon_img,
                    'recon_protos': recon_proto,
                    'dists': dists,
                    's_weights': s_weights,
                    'latent_enc': latent_enc,
                    'proto_vecs': proto_vecs,
                    'agg_protos': agg_proto}
        return res_dict


class Pair_RAE(RAE):
    def __init__(self, input_dim=(1, 1, 28,28), n_z=10, filter_dim=32,
                 n_proto_vecs=(10,),
                 train_pw=False, softmin_temp=1, agg_type='sum',
                 device="cpu"):
        super(Pair_RAE, self).__init__(input_dim=input_dim, n_z=n_z, filter_dim=filter_dim,
                                  n_proto_vecs=n_proto_vecs,
                                  train_pw=train_pw, softmin_temp=softmin_temp, agg_type=agg_type,
                                  device=device)

        self.softmin_temp_pairs = nn.Parameter(torch.ones(softmin_temp,))
        self.softmin_temp_pairs.requires_grad = False

        self.mse = torch.nn.MSELoss()

    def concat_res_dicts(self, res1_single, res2_single):
        """
        Combines the results of the forward passes of each imgs group into a single result dict.
        :param res1_single: forward pass results dict for imgs1
        :param res2_single: forward pass results dict for imgs2
        :return:
        """
        assert res1_single.keys() == res2_single.keys()

        res = {}
        for key in res1_single.keys():
            if key == 'proto_vecs':
                res[key] = self.proto_layer.proto_vecs
            elif isinstance(res1_single[key], dict):
                d = dict()
                for k in range(self.n_proto_groups):
                    d[k] = torch.cat((res1_single[key][k], res2_single[key][k]), dim=0)
                res[key] = d
            elif torch.is_tensor(res1_single[key]):
                res[key] = torch.cat((res1_single[key], res2_single[key]), dim=0)
            else:
                raise ValueError('Unhandled data structure in res1_single, please send email to '
                                 'schramowski@cs.tu-darmstadt.de')
            # store the min weights seprately, not just concatenated
            # first convert to float
            for k in range(self.n_proto_groups):
                res1_single['s_weights'][k] = res1_single['s_weights'][k].float()
                res2_single['s_weights'][k] = res2_single['s_weights'][k].float()
            res['s_weights_pairs'] = [res1_single['s_weights'], res2_single['s_weights']]

        return res

    def comp_dists_sweights(self, s_weights_1, s_weights_2):
        """

        :param s_weights_1: dict length of n_groups for imgs1, each entry has dims [batch, n_protos_in_group],
                            e.g. [64, 4]
        :param s_weights_2: dict length of n_groups for imgs2, each entry has dims [batch, n_protos_in_group],
                            e.g. [64, 4]
        :return:
        """
        # for every group compute the distances between paired s_weights, using l1 norm
        s_weights_dists = torch.empty(s_weights_1[0].shape[0], self.n_proto_groups, device=self.device) # [batch, n_groups]
        for k in range(self.n_proto_groups):
            # compute the l1 norm between the s weight vectors of each img pair
            s_weights_dists[:, k] = ae_helpers.l1_norm(s_weights_1[k], s_weights_2[k])

        # TODO: allow to specify if one attribute is the same of one attribute is different, step2: allow for varying k
        #  step3: profit (please msg schramowski@cs.tu-darmstadt.de for details on this)
        # now perform softmin over the dists between the paired s_weights
        pair_s_weights = self.softmin(self.softmin_temp_pairs * s_weights_dists)

        return pair_s_weights

    def forward_single(self, imgs, std):
        return super().forward(imgs, std)

    def forward(self, img_pair, std=None):
        (imgs1, imgs2) = img_pair

        # pass each individual imgs tensor through forward
        res1_single = self.forward_single(imgs1, std) # returns results dict
        res2_single = self.forward_single(imgs2, std) # returns results dict

        # TODO: enforce the s_weights to be the same for the first group
        # pair_loss = self.mse(res1_single['s_weights'][0], res2_single['s_weights'][0])
        # TODO: just for testing, the prototype weights of one group is set to be the same between imag pairs
        # res2_single['s_weights'][0] = res1_single['s_weights'][0]

        # combine the tensors of both forwards passes to joint result dict
        res = self.concat_res_dicts(res1_single, res2_single)

        # # compute the distances between s weights per group between img pairs
        # pair_s_weights = self.comp_dists_sweights(res1_single['s_weights'],
        #                                           res2_single['s_weights']) # [batch, n_groups]

        res['pair_s_weights'] = []

        return res

