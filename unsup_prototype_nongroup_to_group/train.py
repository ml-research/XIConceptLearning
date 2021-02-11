import torch
import torchvision
import argparse
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import sys
import os
from args import parse_args_as_dict
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from rtpt.rtpt import RTPT
from torch.optim import lr_scheduler

import utils as utils
import losses as losses
from model import RAE
from autoencoder_helpers import list_of_distances
from data import get_dataloader

def train(model, data_loader, log_samples):
    # optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # learning rate scheduler
    if config['lr_scheduler']:
        num_steps = len(data_loader) * config['epochs']
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=2e-5)

    rtpt = RTPT(name_initials='MM', experiment_name='XIC_PrototypeDL', max_iterations=config['epochs'])
    rtpt.start()

    mse = torch.nn.MSELoss()

    for e in range(0, config['epochs']):
        max_iter = len(data_loader)
        start = time.time()
        loss_dict = dict(
            {'img_recon_loss': 0, 'softmin_proto_recon_loss': 0, 'r1_loss': 0, 'r2_loss': 0,
             'r5_loss': 0, 'loss': 0, 'enc_mse_loss': 0, 'ad_loss': 0})

        for i, batch in enumerate(data_loader):
            imgs, _ = map(lambda x: x.to(config['device']), batch)

            std = (config['epochs'] - e) / config['epochs']

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

            softmin_proto_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_softmin_proto'] != 0:
                softmin_proto_recon_loss = mse(rec_protos, imgs)

            img_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_z'] != 0:
                img_recon_loss = mse(rec_imgs, imgs)

            loss_enc_mse = mse(mixed_prototypes, feature_vectors_z.flatten(1,3))

            loss = config['lambda_z'] * img_recon_loss + \
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
            loss_dict['softmin_proto_recon_loss'] += softmin_proto_recon_loss.item()

            loss_dict['r1_loss'] += r1_loss.item()
            loss_dict['r2_loss'] += r2_loss.item()
            loss_dict['enc_mse_loss'] += loss_enc_mse.item()
            loss_dict['ad_loss'] += loss_ad.item()

            loss_dict['loss'] += loss.item()

        for key in loss_dict.keys():
            loss_dict[key] /= len(data_loader)

        rtpt.step(subtitle=f'loss={loss_dict["loss"]:2.2f}')

        if (e + 1) % config['display_step'] == 0 or e == config['epochs'] - 1:
            for key in loss_dict.keys():
                writer.add_scalar(f'train/{key}', loss_dict[key], global_step=e)

        if (e + 1) % config['print_step'] == 0 or e == config['epochs'] - 1:
            print(f'epoch {e} - loss {loss.item():2.4f} - time/epoch {(time.time() - start):2.2f}')
            loss_summary = ''
            for key in loss_dict.keys():
                loss_summary += f'{key} {loss_dict[key]:2.4f} '
            print(loss_summary)

        if (e + 1) % config['save_step'] == 0 or e == config['epochs'] - 1 or e == 0:

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

if __name__ == '__main__':
   
   # get config
    config = parse_args_as_dict(sys.argv[1:])

    # get data
    _data_loader = get_dataloader(config)

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
    # TODO fix add_hparams
    # list_key = []
    # for key in config.keys():
    #     if type(config[key]) is type(list()):
    #         list_key += [key]
                
    # for key in list_key:
    #     for i, item in enumerate(config[key]):
    #         config[key+str(i)] = item
    #     del config[key]

    # writer.add_hparams(config, dict())

    # model setup
    _model = RAE(input_dim=(1, config['img_shape'][0], config['img_shape'][1], config['img_shape'][2]),
                 n_z=config['n_z'], filter_dim=config['filter_dim'],
                 n_prototype_vectors=config['prototype_vectors'],
                 train_pw=config['train_weighted_protos'],
                 device=config['device'])

    _model = _model.to(config['device'])

    # start training
    train(_model, _data_loader, x_set)
