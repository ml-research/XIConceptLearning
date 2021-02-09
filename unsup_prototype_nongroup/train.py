import torch
import torchvision
from torchvision import transforms

import numpy as np

from model import RAE
from network import *
from autoencoder_helpers import makedirs

from data_preprocessing import batch_elastic_transform

import time

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import datetime
from torch.utils.tensorboard import SummaryWriter

from rtpt.rtpt import RTPT

import torchvision.transforms.functional as TF

from PIL import Image

import json

from torch.optim import lr_scheduler

config = dict({
    'device': 'cuda:0',
    
    'save_step': 50,
    'print_step': 10,
    'display_step': 1,

    'mse': True,               # use MSE instead of mean(list_of_norms) for reconstruction loss

    'lambda_min_proto': 0,      # decode protoype with min distance to z
    'lambda_z': 0,              # decode z
    'lambda_softmin_proto': 5,  # decode softmin weighted combination of prototypes
    'lambda_r1': 1e-2,          # draws prototype close to training example
    'lambda_r2': 0,             # draws encoding close to prototype
    'lambda_r3': 0,             # not used
    'lambda_r4': 0,             # not used
    'lambda_r5': 0,             # diversity penalty
    'diversity_threshold': 2,   # 1-2 suggested by paper

    'learning_rate': 1e-3,
    'training_epochs': 500,
    'lr_scheduler': True,
    'batch_size': 1000,
    'n_workers': 2,

    'n_prototype_vectors': 10,
    'filter_dim': 32,
    'n_z': 10,
    'batch_elastic_transform': False,
    'sigma': 4,
    'alpha': 20,

    'experiment_name': '',
    'results_dir': 'results',
    'model_dir': 'states',
    'img_dir': 'imgs',

    'dataset': 'toycolor',       # 'toycolor' or 'mnist' or 'toycolorshape'
    'init': 'xavier',

    'seed': 42
})


def train(model, data_loader):
    # optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # learning rate scheduler
    if config['lr_scheduler']:
        # TODO: try LambdaLR
        num_steps = len(data_loader) * config['training_epochs']
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=2e-5)

    rtpt = RTPT(name_initials='MM', experiment_name='XIC_PrototypeDL', max_iterations=config['training_epochs'])
    rtpt.start()

    for e in range(0, config['training_epochs']):
        max_iter = len(data_loader)
        start = time.time()
        loss_dict = dict({'z_recon_loss':0, 'min_proto_recon_loss':0, 'softmin_proto_recon_loss':0, 'r1_loss':0, 'r2_loss':0, 'r3_loss':0, 'r4_loss':0, 'r5_loss':0, 'loss':0})

        for i, batch in enumerate(data_loader):
            imgs = batch[0]
            # to apply elastic transform, batch has to be flattened
            # store original size to put it back into orginial shape after transformation
            imgs_shape = imgs.shape
            if config['batch_elastic_transform']:
                imgs = batch_elastic_transform(imgs.view(config['batch_size'], -1), sigma=config['sigma'], alpha=config['alpha'], height=config['img_shape'][1], width=config['img_shape'][2])
                imgs = torch.reshape(torch.tensor(imgs), imgs_shape)
            imgs = imgs.to(config['device'])

            labels = batch[1].to(config['device'])

            optimizer.zero_grad()

            pred = model.forward(imgs)

            prototype_vectors = model.prototype_layer.prototype_vectors
            feature_vectors_z = model.enc.forward(imgs)

            # draws prototype close to training example
            r1_loss = torch.mean(torch.min(list_of_distances(prototype_vectors, feature_vectors_z.view(-1, model.input_dim_prototype)), dim=1)[0])
            # draws encoding close to prototype
            r2_loss = torch.mean(torch.min(list_of_distances(feature_vectors_z.view(-1, model.input_dim_prototype ), prototype_vectors), dim=1)[0])

            # experimental
            r3_loss = torch.mean(torch.topk(list_of_distances(feature_vectors_z.view(-1, model.input_dim_prototype ), prototype_vectors), k=2, dim=1, largest=False)[0])
            r4_loss = torch.zeros((1,)).to(config['device'])

            # diversity regularization
            # draw prototypes away from each other
            # Source: Interpretable and Steerable Sequence Learning via Prototypes
            relu = torch.nn.ReLU()
            r5_loss = torch.zeros((1,)).to(config['device'])
            # only compute if needed, cause time you know...
            if config['lambda_r5'] != 0:
                for i in range(prototype_vectors.shape[0]):
                    for j in range(i+1, prototype_vectors.shape[0]):
                        # torch.linalg.norm for torch 1.7.0
                        max_distance = relu(config['diversity_threshold'] - torch.norm(prototype_vectors[i] - prototype_vectors[j]))
                        r5_loss += max_distance**2

            softmin_proto_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_softmin_proto'] != 0:
                softmin = torch.nn.Softmin(dim=1)
                # prototype_vectors_noise = prototype_vectors + torch.normal(torch.zeros_like(prototype_vectors), prototype_vectors.max())
                p_z = list_of_distances(feature_vectors_z.view(-1, model.input_dim_prototype), prototype_vectors, norm='l2')

                std = (config['training_epochs'] - e) / config['training_epochs']
                p_z += torch.normal(torch.zeros_like(p_z), std)

                # experimental cosine similarity
                # ultra slow
                # p_z = torch.zeros((feature_vectors_z.shape[0], prototype_vectors.shape[0])).to(config['device'])
                # for i in range(feature_vectors_z.shape[0]):
                #     for j in range(prototype_vectors.shape[0]):
                #         # print(type(feature_vectors_z[i]), type(prototype_vectors[j]))
                #         cosine = torch.nn.CosineSimilarity()
                #         p_z[i][j] = cosine(feature_vectors_z[i].reshape(1,-1), prototype_vectors[j].unsqueeze(0))

                s = softmin(p_z)
                feature_vectors_softmin = s@prototype_vectors
                # rec_proto = torch.einsum('bi, ij -> bij', s, prototype_vectors)
                rec_proto = model.dec.forward(feature_vectors_softmin.reshape(feature_vectors_z.shape))

                mse = torch.nn.MSELoss()
                if config['mse']:
                    softmin_proto_recon_loss = mse(rec_proto, imgs)
                else:
                    softmin_proto_recon_loss = torch.mean(list_of_norms(rec_proto-imgs))

            z_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_z'] != 0:
                rec = model.dec.forward(feature_vectors_z)
                mse = torch.nn.MSELoss()
                if config['mse']:
                    z_recon_loss = mse(rec, imgs)
                else:
                    z_recon_loss = torch.mean(list_of_norms(rec-imgs))

            # get prototype with min distance to z
            std = (config['training_epochs'] - e) / config['training_epochs']
            pred += torch.normal(torch.zeros_like(pred), std)
            min_prototype_vector = prototype_vectors[torch.argmin(pred, 1)]
            min_proto_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_min_proto'] != 0:
                rec_proto = model.dec.forward(min_prototype_vector.reshape(feature_vectors_z.shape))
                mse = torch.nn.MSELoss()
                if config['mse']:
                    min_proto_recon_loss = mse(rec_proto, imgs)
                else:
                    min_proto_recon_loss = torch.mean(list_of_norms(rec_proto-imgs))

            loss =  config['lambda_z'] * z_recon_loss +\
                    config['lambda_min_proto'] * min_proto_recon_loss +\
                    config['lambda_softmin_proto'] * softmin_proto_recon_loss +\
                    config['lambda_r1'] * r1_loss +\
                    config['lambda_r2'] * r2_loss +\
                    config['lambda_r3'] * r3_loss +\
                    config['lambda_r4'] * r4_loss +\
                    config['lambda_r5'] * r5_loss

            loss.backward()

            optimizer.step()

            if config['lr_scheduler']:
                scheduler.step()

            loss_dict['z_recon_loss'] += z_recon_loss.item()
            loss_dict['min_proto_recon_loss'] += min_proto_recon_loss.item()
            loss_dict['softmin_proto_recon_loss'] += softmin_proto_recon_loss.item()

            loss_dict['r1_loss'] += r1_loss.item()
            loss_dict['r2_loss'] += r2_loss.item()
            loss_dict['r3_loss'] += r3_loss.item()
            loss_dict['r4_loss'] += r4_loss.item()
            loss_dict['r5_loss'] += r5_loss.item()
            loss_dict['loss'] += loss.item()

        for key in loss_dict.keys():
            loss_dict[key] /= len(data_loader)

        rtpt.step(subtitle=f'loss={loss_dict["loss"]:2.2f}')

        if (e+1) % config['display_step'] == 0 or e == config['training_epochs'] - 1:
            for key in loss_dict.keys():
                writer.add_scalar(f'train/{key}', loss_dict[key], global_step=e)

        if (e+1) % config['print_step'] == 0 or e == config['training_epochs'] - 1:
            print(f'epoch {e} - loss {loss.item():2.4f} - time/epoch {(time.time()-start):2.2f}')
            loss_summary = ''
            for key in loss_dict.keys():
                loss_summary += f'{key} {loss_dict[key]:2.4f} '
            print(loss_summary)

        if (e+1) % config['save_step'] == 0 or e == config['training_epochs'] - 1:

            state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ep': e
                    }
            torch.save(state, os.path.join(config['model_dir'], '%05d.pth' % (e)))

            # decode prototype vectors
            prototype_imgs = model.dec.forward(prototype_vectors.reshape((-1,config['n_z'],2,2))).detach().cpu()

            # visualize the prototype images
            n_cols = 3
            n_rows = config['n_prototype_vectors'] // n_cols + 1 if config['n_prototype_vectors'] % n_cols != 0 else config['n_prototype_vectors'] // n_cols
            g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
            for i in range(n_rows):
                for j in range(n_cols):
                    if i*n_cols + j < config['n_prototype_vectors']:
                        b[i][j].imshow(prototype_imgs[i*n_cols + j].reshape(config['img_shape']).permute(1,2,0).squeeze(), # config['img_shape'][1], config['img_shape'][2]
                                        cmap='gray',
                                        interpolation='none')
                        b[i][j].axis('off')

            img_save_path = os.path.join(config['img_dir'], f'{e:05d}' + '_prototype_result' + '.png')
            plt.savefig(img_save_path,
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()

            image = Image.open(img_save_path)
            image = TF.to_tensor(image)
            writer.add_image(f'train/prototype_result', image, global_step=e)

            # apply encoding and decoding over a small subset of the training set
            imgs = []
            for batch in data_loader:
                imgs = batch[0].to(config['device'])
                break

            examples_to_show = 10

            # decoded image
            encoded = model.enc.forward(imgs[:examples_to_show])
            decoded = model.dec.forward(encoded)

            # decoded prototype with min distance to z / image
            pred = model.forward(imgs[:examples_to_show])
            prototype_vectors = model.prototype_layer.prototype_vectors
            min_prototype_vector = prototype_vectors[torch.argmin(pred, 1)]
            decoded_proto_min = model.dec.forward(min_prototype_vector.reshape(encoded.shape))

            # decoded softmin weighted prototype
            softmin = torch.nn.Softmin(dim=1)
            p_z = list_of_distances(encoded.view(-1, model.input_dim_prototype), prototype_vectors, norm='l2')
            # std = (config['training_epochs'] - e) / config['training_epochs']
            # p_z += torch.normal(torch.zeros_like(p_z), std)
            s = softmin(p_z)
            softmin_prototype_vector = s@prototype_vectors

            decoded_proto_softmin = model.dec.forward(softmin_prototype_vector.reshape(encoded.shape))

            decoded = decoded.detach().cpu()
            decoded_proto_min = decoded_proto_min.detach().cpu()
            decoded_proto_softmin = decoded_proto_softmin.detach().cpu()
            imgs = imgs.detach().cpu()

            # compare original images to their reconstructions
            n_rows = 4
            f, a = plt.subplots(n_rows, examples_to_show, figsize=(examples_to_show, n_rows))

            a[0][0].text(0,-2, s='input', fontsize=10)
            a[1][0].text(0,-2, s='recon z', fontsize=10)
            a[2][0].text(0,-2, s='recon min proto', fontsize=10)
            a[3][0].text(0,-2, s='recon softmin proto', fontsize=10)

            for i in range(examples_to_show):
                a[0][i].imshow(imgs[i].reshape(config['img_shape']).permute(1,2,0).squeeze(),
                                cmap='gray',
                                interpolation='none')
                a[0][i].axis('off')

                a[1][i].imshow(decoded[i].reshape(config['img_shape']).permute(1,2,0).squeeze(),
                                cmap='gray',
                                interpolation='none')
                a[1][i].axis('off')

                a[2][i].imshow(decoded_proto_min[i].reshape(config['img_shape']).permute(1,2,0).squeeze(),
                                cmap='gray',
                                interpolation='none')
                a[2][i].axis('off')

                a[3][i].imshow(decoded_proto_softmin[i].reshape(config['img_shape']).permute(1,2,0).squeeze(),
                                cmap='gray',
                                interpolation='none')
                a[3][i].axis('off')

            img_save_path = os.path.join(config['img_dir'], f'{e:05d}' + '_decoding_result' + '.png')
            plt.savefig(img_save_path,
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()

            image = Image.open(img_save_path)
            image = TF.to_tensor(image)
            writer.add_image(f'train/decoding_result', image, global_step=e)

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
        config['n_prototype_vectors'] = train_labels.shape[1]
        print('Overriding img_shape and n_prototype_vectors')

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
    if config['experiment_name'] == '':
        config['experiment_name'] = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]

    config['results_dir'] = os.path.join(config['results_dir'], config['experiment_name'])
    config['model_dir'] = os.path.join(config['results_dir'], config['model_dir'])
    config['img_dir'] = os.path.join(config['results_dir'], config['img_dir'])
    makedirs(config['model_dir'])
    makedirs(config['img_dir'])

    torch.manual_seed(config['seed'])

    _data_loader = init_dataset()

    writer = SummaryWriter(log_dir=config['results_dir'])

    # store config
    with open(os.path.join(config['results_dir'], 'args.json'), 'w') as json_file:
        json.dump(config, json_file, indent=2)

    # model setup
    _model = RAE(input_dim=(1, config['img_shape'][0], config['img_shape'][1], config['img_shape'][2]),
                n_z=config['n_z'], filter_dim=config['filter_dim'], n_prototype_vectors=config['n_prototype_vectors'])
    _model = _model.to(config['device'])
    train(_model, _data_loader)