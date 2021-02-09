import torch
import torch.nn as nn
import autoencoder_helpers as ae_helpers


class RAE(nn.Module):
    def __init__(self, input_dim=(1, 1, 28, 28),
                 n_prototype_groups=2, n_prototype_vectors_per_group=5):
        super(RAE, self).__init__()

        self.n_prototype_groups = n_prototype_groups
        self.n_prototype_vectors_per_group = n_prototype_vectors_per_group

        # encoder
        self.enc = Encoder()

        # prototype layer
        # forward encoder to determine input dim for prototype layer
        self.enc_out = self.enc.forward(torch.randn(input_dim))
        self.input_dim_prototype = self.enc_out.view(-1, 1).shape[0]
        self.prototype_layer = PrototypeLayer(input_dim=self.input_dim_prototype,
                                              n_prototype_groups=n_prototype_groups,
                                              n_prototype_vectors_per_group=n_prototype_vectors_per_group
                                              )

        # decoder
        self.dec_img = Decoder()
        self.dec_proto = Decoder()

        # TODO: the learnable mixture function, ie. learns the combination of different attribute prototypes
        #  -> make this more complex, e.g. some form of attention?
        # first naive approach: stack the prototypes from each group and apply a nonlinear mapping to the encoding space
        self.prototype_mixture_func = nn.Sequential(
            nn.Linear(self.input_dim_prototype * self.n_prototype_groups, self.input_dim_prototype),
            nn.ReLU()
        )

        self.softmin = nn.Softmin(dim=1)

    # compute per group of prototypes which subprototype is nearest to the image encoding
    def prototype_per_group(self, proto_dists, batch_size, noise_std):
        # add gaussian noise to distance measurement
        proto_dists = [proto_dists[i] + torch.normal(torch.zeros_like(proto_dists[i]), noise_std)
                       for i in range(self.n_prototype_groups)]

        # compute per group the weights s, i.e. for every attribute prototype in a group measure the
        # similarity to the image encoding
        # s lies in [0, 1]
        s = [self.softmin(proto_dists[i]) for i in range(self.n_prototype_groups)]
        # recombine a weighted prototype per group
        per_group_prototype = [s[i] @ self.prototype_layer.prototype_vectors[i, :, :] for i in
                               range(self.n_prototype_groups)]
        # stack and reshape the prototypes to get [batch, n_groups, prototype_length]
        per_group_prototype = torch.stack(per_group_prototype).reshape(
            (batch_size, self.n_prototype_groups, self.input_dim_prototype))

        return per_group_prototype

    def mix_prototype_per_sample(self, per_group_prototype, batch_size):
        # TODO: when using more complex mixture function, do not flatten dims
        # for linear mixture function, flatten the n_groups dim
        per_group_prototype = per_group_prototype.reshape(
            (batch_size, per_group_prototype.shape[1] * per_group_prototype.shape[2]))

        # apply mixture function on the weighted prototype of each group to receive a prototype for each sample of the
        # training batch that is composed of the nearest prototypes of each attribute group
        proto_latent = self.prototype_mixture_func(per_group_prototype)

        return proto_latent

    def dec_prototype(self, proto_latent, latent_shape):
        return self.dec_proto(proto_latent.reshape(latent_shape))

    def forward(self, x, noise_std=0.001):
        img_latent = self.enc(x)

        # compute per group of prototypes which subprototype is nearest to the image encoding
        proto_dists = self.prototype_layer(img_latent.view(-1, self.input_dim_prototype))

        # compute the weighted prototype per group for each sample, based on the distances, proto_dists
        per_group_prototype = self.prototype_per_group(proto_dists, batch_size=x.shape[0], noise_std=noise_std)

        # mix the prototypes per group to one
        proto_latent = self.mix_prototype_per_sample(per_group_prototype, batch_size=x.shape[0])

        # reshape to original latent shape, decode
        rec_proto = self.dec_prototype(proto_latent, img_latent.shape)

        # reconstruct the image
        rec_img = self.dec_img(img_latent)

        return rec_img, rec_proto, proto_latent, img_latent, per_group_prototype


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

    def forward(self, x):
        out = self.model(x)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.model(x)
        return out


# TODO: make distance computing between attribute prototype and img encoding more complex, e.g. via attention?
class PrototypeLayer(nn.Module):
    def __init__(self, input_dim=10, n_prototype_groups=2, n_prototype_vectors_per_group=5):
        super(PrototypeLayer, self).__init__()

        self.n_prototype_groups = n_prototype_groups
        self.prototype_vectors = torch.nn.Parameter(torch.rand(n_prototype_groups, n_prototype_vectors_per_group,
                                                               input_dim))
        nn.init.xavier_uniform_(self.prototype_vectors, gain=1.0)

    def forward(self, x):
        distances_per_group = [ae_helpers.list_of_distances(x, self.prototype_vectors[k, :, :], norm='l2')
                               for k in range(self.n_prototype_groups)]
        return distances_per_group


if __name__ == "__main__":
    from torchsummary import summary

    config = dict({
        'device': 'cuda:13',

        'save_step': 50,
        'print_step': 10,
        'display_step': 1,

        'mse': True,  # use MSE instead of mean(list_of_norms) for reconstruction loss

        'lambda_min_proto': 0,  # decode protoype with min distance to z
        'lambda_z': 0,  # decode z
        'lambda_softmin_proto': 5,  # decode softmin weighted combination of prototypes
        'lambda_r1': 1e-2,  # draws prototype close to training example
        'lambda_r2': 0,  # draws encoding close to prototype
        'lambda_r3': 0,  # not used
        'lambda_r4': 0,  # not used
        'lambda_r5': 0,  # diversity penalty
        'diversity_threshold': 2,  # 1-2 suggested by paper

        'learning_rate': 1e-3,
        'training_epochs': 500,
        'lr_scheduler': True,
        'batch_size': 1000,
        'n_workers': 2,

        'img_shape': (3, 28, 28),

        'n_prototype_groups': 3,
        'n_prototype_vectors_per_group': 5,
        # 'filter_dim': 32,
        # 'n_z': 10,
        'batch_elastic_transform': False,
        'sigma': 4,
        'alpha': 20,

        'experiment_name': '',
        'results_dir': 'results',
        'model_dir': 'states',
        'img_dir': 'imgs',

        'dataset': 'mnist',  # 'toycolor' or 'mnist' or 'toycolorshape'
        'init': 'xavier'
    })
    net = RAE(input_dim=(1, config['img_shape'][0], config['img_shape'][1], config['img_shape'][2]),
              n_prototype_groups=config['n_prototype_groups'],
              n_prototype_vectors_per_group=config['n_prototype_vectors_per_group'])
    # net = ConvAutoencoder()

    # summary(net, (3, 28, 28))
    batch = torch.rand((15, 3, 28, 28))
    out = net(batch)
