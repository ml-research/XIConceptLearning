import torch
import torch.nn as nn
import modules as modules
import autoencoder_helpers as ae_helpers

class RAE(nn.Module):
    def __init__(self, input_dim=(1, 1, 28,28), n_z=10, filter_dim=32,
                 n_proto_vecs=(10,),
                 train_pw=False, softmax_temp=0.01, agg_type='sum',
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
        self.latent_dim_flat = self.enc_out.view(-1,1).shape[0]
        self.dim_proto = self.latent_dim_flat

        # prototype layer
        self.proto_layer = modules.PrototypeLayer(input_dim=self.dim_proto,
                                                  n_proto_vecs=self.n_proto_vecs,
                                                  device=self.device)

        self.proto_agg_layer = modules.ProtoAggregateLayer(n_protos=self.n_proto_groups,
                                                           dim_protos=self.dim_proto,
                                                           train_pw=train_pw,
                                                           layer_type=agg_type,
                                                           device=self.device)

        # decoder
        # use forwarded encoder to determine output shapes for decoder
        dec_out_shapes = []
        for module in self.enc.modules():
            if isinstance(module, modules.ConvLayer):
                dec_out_shapes += [list(module.in_shape)]
        self.dec = modules.Decoder(input_dim=n_z, filter_dim=filter_dim,
                                   output_dim=input_dim[1], out_shapes=dec_out_shapes)
        self.dec_proto = modules.Decoder(input_dim=n_z, filter_dim=filter_dim,
                                   output_dim=input_dim[1], out_shapes=dec_out_shapes)

        self.attr_predictors = modules.AttributePredictors(in_dim=self.latent_dim_flat,
                                                           n_proto_vecs=self.n_proto_vecs,
                                                           temp=softmax_temp,
                                                           device=self.device)

    def comp_combined_prototype_per_sample(self, attr_ids):
        """
        Given the softmin weighted prototypes per group, compute the mixture of these prototypes to combine to one
        prototype.
        :param prototype_vectors_softmin:
        :return:
        """
        # make sure not empty dict
        assert attr_ids
        assert isinstance(attr_ids, dict)

        # extract those attribute slots that were predicted into a tensor
        pred_proto_vecs = torch.empty((len(attr_ids[0]), self.n_proto_groups, self.dim_proto), device=self.device) # [batch, n_groups, dim_proto]
        for k in range(self.n_proto_groups):
            pred_proto_vecs[:, k, :] = self.proto_layer.proto_vecs[k][attr_ids[k], :]

        out = self.proto_agg_layer(pred_proto_vecs)
        return out

    def dec_wrapper(self, enc, Proto=False):
        """
        Wrapper helper for decoding, helpful due to reshaping.
        :param enc:
               Proto:
        :return:
        """
        if Proto:
            return self.dec_proto(enc.reshape([enc.shape[0]] + self.latent_shape))
        else:
            return self.dec(enc.reshape([enc.shape[0]] + self.latent_shape))
    # def dec_wrapper(self, enc):
    #     """
    #     Wrapper helper for decoding, helpful due to reshaping.
    #     :param enc:
    #     :return:
    #     """
    #     return self.dec(enc.reshape([enc.shape[0]] + self.latent_shape))

    def forward_decoder(self, latent_enc, agg_proto):
        # decode mixed prototypes
        recon_proto = self.dec_wrapper(agg_proto, Proto=True)

        # decode latent encoding
        recon_img = self.dec_wrapper(latent_enc, Proto=False)

        return recon_img, recon_proto

    def forward_encoder(self, x):
        return self.enc(x)

    def forward(self, x):
        # x: [batch, ch, w, h]

        latent_enc = self.forward_encoder(x) # [batch, latent_ch, latent_w, latent_h]

        # predict from encoding which attribute is present
        attr_prob = self.attr_predictors(latent_enc) # dict [n_groups, batch]

        # apply argmax over predicitons
        attr_ids = dict()
        for k in range(self.n_proto_groups):
            attr_ids[k] = torch.argmax(attr_prob[k], dim=1)

        # combine those attributes that were predicted to a single vector
        agg_proto = self.comp_combined_prototype_per_sample(attr_ids)

        # decode the image and the prototypes
        recon_img, recon_proto = self.forward_decoder(latent_enc, agg_proto)

        res_dict = self.create_res_dict(recon_img, recon_proto, attr_prob, latent_enc,
                                        self.proto_layer.proto_vecs, agg_proto)

        return res_dict

    def create_res_dict(self, recon_img, recon_proto, attr_prob, latent_enc, proto_vecs, agg_proto):
        res_dict = {'recon_imgs': recon_img,
                    'recon_protos': recon_proto,
                    'attr_prob': attr_prob,
                    'latent_enc': latent_enc,
                    'proto_vecs': proto_vecs,
                    'agg_protos': agg_proto}
        return res_dict


class Pair_RAE(RAE):
    def __init__(self, input_dim=(1, 1, 28,28), n_z=10, filter_dim=32,
                 n_proto_vecs=(10,),
                 train_pw=False, softmax_temp=1, agg_type='sum',
                 device="cpu"):
        super(Pair_RAE, self).__init__(input_dim=input_dim, n_z=n_z, filter_dim=filter_dim,
                                  n_proto_vecs=n_proto_vecs,
                                  train_pw=train_pw, softmax_temp=softmax_temp, agg_type=agg_type,
                                  device=device)

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
        # store the dists seprately, not just concatenated
        res['attr_prob_pairs'] = [res1_single['attr_prob'], res2_single['attr_prob']]

        return res

    def forward_single(self, imgs):
        return super().forward(imgs)

    def forward(self, img_pair):
        (img1, img2) = img_pair

        # pass each individual imgs tensor through forward
        res1_single = self.forward_single(img1) # returns results dict
        res2_single = self.forward_single(img2) # returns results dict

        # combine the tensors of both forwards passes to joint result dict
        res = self.concat_res_dicts(res1_single, res2_single)

        return res


if __name__ == "__main__":
    from torchsummary import summary
    net = Pair_RAE(input_dim=(1, 3, 28, 28),
                 n_z=10, filter_dim=32,
                 n_proto_vecs=[4, 2],
                 train_pw=False,
                 softmax_temp=0.01,
                 device='cpu',
                 agg_type='sum')

    # summary(net, (3, 28, 28))
    # x = torch.rand(15, 3, 28, 28)
    # out = net(x)
    x1 = torch.rand(15, 3, 28, 28)
    x2 = torch.rand(15, 3, 28, 28)
    out = net((x1, x2))
