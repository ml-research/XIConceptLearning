import torch
import torch.nn as nn
import autoencoder_helpers as ae_helpers


class RAE(nn.Module):
    def __init__(self, input_dim=(1, 1, 28, 28),
                 n_prototype_groups=2, n_prototype_vectors_per_group=5,
                 device="cpu"):
        super(RAE, self).__init__()

        self.n_prototype_groups = n_prototype_groups
        self.n_prototype_vectors_per_group = n_prototype_vectors_per_group
        self.device = device

        # encoder
        self.enc = Encoder()

        # prototype layer
        # forward encoder to determine input dim for prototype layer
        self.enc_out = self.enc.forward(torch.randn(input_dim))
        self.input_dim_prototype = self.enc_out.view(-1, 1).shape[0]

        # TODO: the learnable mixture function, ie. learns the combination of different attribute prototypes
        #  -> make this more complex, e.g. some form of attention?
        # first naive approach: stack the prototypes from each group and apply a nonlinear mapping to the encoding space
        self.prototype_mixture_func = nn.Sequential(
            nn.Linear(self.input_dim_prototype * self.n_prototype_groups, self.input_dim_prototype),
            nn.ReLU()
        )

        self.prototype_layer = PrototypeLayer(mixture_fn=self.prototype_mixture_func,
                                              input_dim=self.input_dim_prototype,
                                              n_prototype_groups=n_prototype_groups,
                                              n_prototype_vectors_per_group=n_prototype_vectors_per_group,
                                              device=self.device
                                              )

        # decoder
        self.dec_img = Decoder()
        self.dec_proto = Decoder()

        self.softmin = nn.Softmin(dim=1)

    def comp_weighted_prototype(self, proto_dists, noise_std):
        """
        Weights each mixed prototype by the softmin weight given the distances to the image encoding.
        :param proto_dists:
        :param noise_std:
        :return:
        """
        # add gaussian noise to distance measurement
        proto_dists = proto_dists + torch.normal(torch.zeros_like(proto_dists), noise_std)

        # compute the weights s; s lies in [0, 1]
        s = self.softmin(proto_dists)

        # compute a weighted prototype
        weighted_prototype = s @ self.prototype_layer.mixed_prototype_vectors

        return weighted_prototype

    def dec_prototype(self, proto_latent, latent_shape):
        """
        Wrapper for prototype decoding.
        :param proto_latent:
        :param latent_shape:
        :return:
        """
        return self.dec_proto(proto_latent.reshape(latent_shape))

    def forward(self, x, noise_std=0.001):
        img_latent = self.enc(x)

        # compute per group of prototypes which subprototype is nearest to the image encoding
        proto_dists = self.prototype_layer(img_latent.view(-1, self.input_dim_prototype))

        # compute the weighted prototype per group for each sample, based on the distances, proto_dists
        proto_latent = self.comp_weighted_prototype(proto_dists, noise_std=noise_std)

        # reshape to original latent shape, decode
        rec_proto = self.dec_prototype(proto_latent, img_latent.shape)

        # reconstruct the image
        rec_img = self.dec_img(img_latent)

        return rec_img, rec_proto, proto_latent, img_latent


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
    def __init__(self, mixture_fn, input_dim=10, n_prototype_groups=2, n_prototype_vectors_per_group=5, device="cpu"):
        super(PrototypeLayer, self).__init__()

        self.mixture_fn = mixture_fn
        self.input_dim = input_dim
        self.n_prototype_groups = n_prototype_groups
        self.n_prototype_vectors_per_group = n_prototype_vectors_per_group
        self.device = device
        self.mixture_fn.to(self.device)

        self.prototype_vectors = torch.nn.Parameter(torch.rand(n_prototype_groups, n_prototype_vectors_per_group,
                                                               input_dim, device=self.device),
                                                    requires_grad=True)
        nn.init.xavier_uniform_(self.prototype_vectors, gain=1.0)

        self.mixed_prototype_vectors = torch.empty((self.n_prototype_vectors_per_group ** self.n_prototype_groups,
                                                self.input_dim), device=self.device, requires_grad=True)
        self.update_mixed_prototypes() # initialize global prototypes

    def update_mixed_prototypes(self):
        """
        self.n_prototype_vectors_per_group gets updated with each backpropagation pass, therefore we must update the
        mixed prototype vectors, i.e. recompute the mixtures.
        :return:
        """
        mixed_prototype_vectors = torch.empty((self.n_prototype_vectors_per_group ** self.n_prototype_groups,
                                               self.input_dim),
                                              device=self.device,
                                              requires_grad=False)
        for j in range(self.n_prototype_vectors_per_group):
            for l in range(self.n_prototype_vectors_per_group):
                mixed_prototype_vectors[j*self.n_prototype_vectors_per_group + l, :] = \
                    self.mixture_fn(torch.stack((self.prototype_vectors[0, j], self.prototype_vectors[1, l])).view(-1))
        return mixed_prototype_vectors

    def forward(self, x):
        """
        First update the list of mixed prototypes, given the individual grouped prototypes. Then compute the distance
        between the encoded image and these mixed prototypes.
        :param x:
        :return:
        """
        self.mixed_prototype_vectors = self.update_mixed_prototypes()
        return ae_helpers.list_of_distances(x, self.mixed_prototype_vectors)


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

        'n_prototype_groups': 2,
        'n_prototype_vectors_per_group': 4,
        'filter_dim': 32,
        'n_z': 10,
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
    sample = torch.rand((15, 3, 28, 28))
    out = net(sample)