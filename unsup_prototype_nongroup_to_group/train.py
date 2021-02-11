import torch
import torchvision
import argparse
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import os
import datetime
import json
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from rtpt.rtpt import RTPT
from torch.optim import lr_scheduler

import utils as utils
import losses as losses
from model import RAE
from autoencoder_helpers import makedirs, list_of_distances

# TODO: convert config fully to argparse?
config = dict({
    'device': 'cuda',
    # 'device': 'cpu',

    'save_step': 50,
    'print_step': 10,
    'display_step': 1,

    'mse': True,  # use MSE instead of mean(list_of_norms) for reconstruction loss

    'lambda_min_proto': 0,  # decode protoype with min distance to z
    'lambda_z': 0,  # decode z
    'lambda_softmin_proto': 5,  # decode softmin weighted combination of prototypes
    'lambda_r1': 1e-2,  # draws prototype close to training example
    'lambda_r2': 0, #1e-2,  # draws encoding close to prototype
    'lambda_enc_mse': 0,
    'lambda_ad': 1e-2,
    'diversity_threshold': 2,  # 1-2 suggested by paper
    'train_weighted_protos': False,

    'learning_rate': 1e-3,
    'lr_scheduler': True,
    'batch_size': 1000,
    'n_workers': 2,

    'n_prototype_vectors': 4,
    'n_prototype_groups': 2,
    'filter_dim': 32,
    'n_z': 10,
    'batch_elastic_transform': False,
    'sigma': 4,
    'alpha': 20,

    'experiment_name': '',
    'results_dir': 'results_group',
    'model_dir': 'states',
    'img_dir': 'imgs',

    'dataset': 'toycolor',  # 'toycolor' or 'mnist' or 'toycolorshape'
    'init': 'xavier',
})


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--seed", type=int, default=0, help="seed"
    )
    parser.add_argument(
        "-p", "--n_prototype_vectors", type=int, default=4, help="num of prototypes"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=500, help="num of epochs to train"
    )
    parser.add_argument(
        "-pg", "--prototype-groups", type=int, default=2, help="num prototype groups"
    )

    args = parser.parse_args()
    return args


def train(model, data_loader, log_samples):
    # optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # learning rate scheduler
    if config['lr_scheduler']:
        # TODO: try LambdaLR
        num_steps = len(data_loader) * config['training_epochs']
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=2e-5)

    rtpt = RTPT(name_initials='MM', experiment_name='XIC_PrototypeDL', max_iterations=config['training_epochs'])
    rtpt.start()

    mse = torch.nn.MSELoss()

    for e in range(0, config['training_epochs']):
        max_iter = len(data_loader)
        start = time.time()
        loss_dict = dict(
            {'img_recon_loss': 0, 'min_proto_recon_loss': 0, 'softmin_proto_recon_loss': 0, 'r1_loss': 0, 'r2_loss': 0,
             'r5_loss': 0, 'loss': 0, 'enc_mse_loss': 0, 'ad_loss': 0})

        for i, batch in enumerate(data_loader):
            imgs, _ = map(lambda x: x.to(config['device']), batch)

            # TODO: can we just remove the lines to batch_elastic, if we never use it?
            # # to apply elastic transform, batch has to be flattened
            # # store original size to put it back into orginial shape after transformation
            # imgs_shape = imgs.shape
            # if config['batch_elastic_transform']:
            #     imgs = batch_elastic_transform(imgs.view(config['batch_size'], -1), sigma=config['sigma'],
            #                                    alpha=config['alpha'], height=config['img_shape'][1],
            #                                    width=config['img_shape'][2])
            #     imgs = torch.reshape(torch.tensor(imgs), imgs_shape)

            std = (config['training_epochs'] - e) / config['training_epochs']

            rec_imgs, rec_protos, dists, feature_vectors_z, prototype_vectors, mixed_prototypes = model.forward(imgs, std)

            # draws prototype close to training example
            r1_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_r1'] != 0:
                r1_loss = losses.r1_loss(prototype_vectors, feature_vectors_z, model.dim_prototype, config)

            # draws encoding close to prototype
            r2_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_r2'] != 0:
                r2_loss = losses.r2_loss(prototype_vectors, feature_vectors_z, model.dim_prototype, config)

            loss_ad = torch.zeros((1,)).to(config['device'])

            if config['lambda_ad'] != 0:
                for k in range(len(prototype_vectors)):
                    loss_ad += torch.mean(torch.sqrt(torch.sum(prototype_vectors[k].T ** 2, dim=1)), dim=0)

            # TODO: can we remove this?
            # # diversity regularization
            # # draw prototypes away from each other
            # # Source: Interpretable and Steerable Sequence Learning via Prototypes
            # relu = torch.nn.ReLU()
            # r5_loss = torch.zeros((1,)).to(config['device'])
            # # only compute if needed, cause time you know...
            # if config['lambda_r5'] != 0:
            #     for i in range(prototype_vectors.shape[1]):
            #         for j in range(i + 1, prototype_vectors.shape[1]):
            #             # torch.linalg.norm for torch 1.7.0
            #             max_distance = relu(
            #                 config['diversity_threshold'] - torch.norm(prototype_vectors[i] - prototype_vectors[j]))
            #             r5_loss += max_distance ** 2

            softmin_proto_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_softmin_proto'] != 0:
                softmin_proto_recon_loss = mse(rec_protos, imgs)

            img_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_z'] != 0:
                img_recon_loss = mse(rec_imgs, imgs)

            # TODO: do we need this?
            # get prototype with min distance to z
            min_proto_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_min_proto'] != 0:
                min_prototype_vectors = prototype_vectors[torch.argmin(dists, 2)] # [batch, n_groups]
                # TODO: we now have to handle that there are one min prototype per group --> maybe move to model?
                # rec_proto = model.dec.forward(min_prototype_vector.reshape(feature_vectors_z.shape))
                # mse = torch.nn.MSELoss()
                # min_proto_recon_loss = mse(rec_proto, imgs)

            loss_enc_mse = mse(mixed_prototypes, feature_vectors_z.flatten(1,3))

            loss = config['lambda_z'] * img_recon_loss + \
                   config['lambda_min_proto'] * min_proto_recon_loss + \
                   config['lambda_softmin_proto'] * softmin_proto_recon_loss + \
                   config['lambda_r1'] * r1_loss + \
                   config['lambda_r2'] * r2_loss + \
                   config['lambda_enc_mse'] * loss_enc_mse + \
                   config['lambda_ad'] * loss_ad

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config['lr_scheduler']:
                scheduler.step()

            loss_dict['img_recon_loss'] += img_recon_loss.item()
            loss_dict['min_proto_recon_loss'] += min_proto_recon_loss.item()
            loss_dict['softmin_proto_recon_loss'] += softmin_proto_recon_loss.item()

            loss_dict['r1_loss'] += r1_loss.item()
            loss_dict['r2_loss'] += r2_loss.item()
            loss_dict['enc_mse_loss'] += loss_enc_mse.item()
            loss_dict['ad_loss'] += loss_ad.item()

            loss_dict['loss'] += loss.item()

        for key in loss_dict.keys():
            loss_dict[key] /= len(data_loader)

        rtpt.step(subtitle=f'loss={loss_dict["loss"]:2.2f}')

        if (e + 1) % config['display_step'] == 0 or e == config['training_epochs'] - 1:
            for key in loss_dict.keys():
                writer.add_scalar(f'train/{key}', loss_dict[key], global_step=e)

        if (e + 1) % config['print_step'] == 0 or e == config['training_epochs'] - 1:
            print(f'epoch {e} - loss {loss.item():2.4f} - time/epoch {(time.time() - start):2.2f}')
            loss_summary = ''
            for key in loss_dict.keys():
                loss_summary += f'{key} {loss_dict[key]:2.4f} '
            print(loss_summary)

        if (e + 1) % config['save_step'] == 0 or e == config['training_epochs'] - 1 or e == 0:

            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ep': e,
                'config': config
            }
            torch.save(state, os.path.join(config['model_dir'], '%05d.pth' % (e)))

            # plot the indivisual prototypes of each group
            utils.plot_prototypes(model, prototype_vectors, writer, e, config)

            # plot a few samples with proto recon
            utils.plot_examples(log_samples, model, writer, e, config)

            print(f'SAVED - epoch {e} - imgs @ {config["img_dir"]} - model @ {config["model_dir"]}')


def init_dataset():
    # dataloader setup
    if config['dataset'] == 'mnist':
        mnist_data = torchvision.datasets.MNIST(root='dataset', train=True, download=True,
                                                transform=transforms.ToTensor())
        mnist_data_test = torchvision.datasets.MNIST(root='dataset', train=False, download=True,
                                                     transform=transforms.ToTensor())

        config['img_shape'] = (1, 28, 28)
        config['n_prototype_vectors'] = 10
        print('Overriding img_shape and n_prototype_vectors')

        dataset = torch.utils.data.ConcatDataset((mnist_data, mnist_data_test))

    elif config['dataset'] == 'toycolor':
        train_data = np.load('data/train_toydata.npy')
        train_data = torch.Tensor(train_data).permute(0, 3, 1, 2)
        train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

        train_labels = np.load('data/train_toydata_labels.npy')
        train_labels = torch.Tensor(train_labels)

        config['img_shape'] = (3, 28, 28)
        # config['n_prototype_vectors'] = train_labels.shape[1]
        print('Overriding img_shape')

        dataset = torch.utils.data.TensorDataset(train_data, train_labels)

    elif config['dataset'] == 'toycolorshape':
        train_data = np.load('data/train_toydata_color_shape.npy')
        train_data = torch.Tensor(train_data).permute(0, 3, 1, 2)
        train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

        train_labels = np.load('data/train_toydata_color_shape_labels.npy')
        train_labels = torch.Tensor(train_labels)

        config['img_shape'] = (3, 28, 28)
        config['n_prototype_vectors'] = train_labels.shape[1]
        print('Overriding img_shape and n_prototype_vectors')

        dataset = torch.utils.data.TensorDataset(train_data, train_labels)

    else:
        raise ValueError('Select valid dataset please: mnist, toycolor')

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                                              num_workers=config['n_workers'], pin_memory=True)

    return data_loader


if __name__ == '__main__':
    _args = get_args()
    config['seed'] = _args.seed
    config['n_prototype_vectors'] = _args.n_prototype_vectors
    config['training_epochs'] = _args.epochs
    config['n_prototype_groups'] = _args.prototype_groups
    if config['experiment_name'] == '':
        config['experiment_name'] = 'seed' + str(config['seed']) + '_' \
                                    + 'protos' + str(config['n_prototype_vectors']) + '_' + \
                                    str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]

    config['results_dir'] = os.path.join(config['results_dir'], config['experiment_name'])
    config['model_dir'] = os.path.join(config['results_dir'], config['model_dir'])
    config['img_dir'] = os.path.join(config['results_dir'], config['img_dir'])
    makedirs(config['model_dir'])
    makedirs(config['img_dir'])

    # set seed for all random processes
    utils.set_seed(config['seed'])

    # get data
    _data_loader = init_dataset()

    # generate set of all individual samples
    x = _data_loader.dataset.tensors[0].detach().numpy().tolist()
    y = _data_loader.dataset.tensors[1].detach().numpy().tolist()
    y_set = np.unique(y, axis=0).tolist()
    x_set = []
    for u in y_set:
        x_set.append(x[y.index(u)])
    x_set = torch.Tensor(x_set)
    x_set = x_set.to(config['device'])

    # create tb writer
    writer = SummaryWriter(log_dir=config['results_dir'])

    # store config
    with open(os.path.join(config['results_dir'], 'args.json'), 'w') as json_file:
        json.dump(config, json_file, indent=2)

    # model setup
    _model = RAE(input_dim=(1, config['img_shape'][0], config['img_shape'][1], config['img_shape'][2]),
                 n_z=config['n_z'], filter_dim=config['filter_dim'],
                 n_prototype_vectors=config['n_prototype_vectors'],
                 n_prototype_groups=config['n_prototype_groups'],
                 train_pw=config['train_weighted_protos'],
                 device=config['device'])

    _model = _model.to(config['device'])

    # start training
    train(_model, _data_loader, x_set)
