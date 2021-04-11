import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib

matplotlib.use('Agg')
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from rtpt.rtpt import RTPT
from torch.optim import lr_scheduler

import VQ_VAE_selfsup.utils as utils
import VQ_VAE_selfsup.data as data
from VQ_VAE_selfsup.models.gproto_ae_selfsup import GProtoAETriplet
from VQ_VAE_selfsup.args import parse_args_as_dict


def train(model, data_loader, log_samples, optimizer, scheduler, writer, config):

    model.train()

    rtpt = RTPT(name_initials=config['initials'], experiment_name='XIC_PrototypeDL', max_iterations=config['epochs'])
    rtpt.start()

    cls_criterion = nn.CrossEntropyLoss()
    for e in range(0, config['epochs']):
        max_iter = len(data_loader)
        start = time.time()
        loss_dict = dict(
            {'recon_loss_z': 0, 'recon_loss_proto': 0, 'loss': 0, 'vq_loss': 0, 'triplet_loss': 0})

        # torch.autograd.set_detect_anomaly(True)
        for i, batch in enumerate(data_loader):
            imgs0, aug_imgs0, imgs1 = batch[0]
            imgs0 = imgs0.to(config['device'])
            aug_imgs0 = aug_imgs0.to(config['device'])
            imgs1 = imgs1.to(config['device'])
            imgs = (imgs0, aug_imgs0, imgs1)

            # labels0, labels1 = batch[1]
            # # reshape to have prediction for each attribute
            # labels0 = labels0.reshape(-1, config['n_groups'], config['n_protos']).permute(1, 0, 2)
            # labels1 = labels1.reshape(-1, config['n_groups'], config['n_protos']).permute(1, 0, 2)
            # # TODO: hack because we don'' have custom dataloader, we convert one hot to class id
            # labels0 = torch.argmax(labels0, dim=2) # [B, G]
            # labels1 = torch.argmax(labels1, dim=2) # [B, G]
            # shared_labels = (labels0 == labels1)
            # labels = torch.ones(labels0.shape, dtype=torch.int64) * -1
            # labels[shared_labels] = labels0[shared_labels]
            # labels = labels.to(config['device'])

            std = (config['epochs'] - e) / config['epochs'] * 0.1

            imgs_recon_z, distances, triplet_loss = model.forward_triplet_cont(imgs, k=1)

            # plot training images
            if i == 0 and ((e + 1) % config['save_step'] == 0 or e == config['epochs'] - 1 or e == 0):
                utils.plot_train_examples(imgs, distances, writer, config, step=e)

            # # reconstruciton loss for z
            # recon_loss_z0 = F.mse_loss(imgs_recon_z[0], imgs0)
            # recon_loss_z1 = F.mse_loss(imgs_recon_z[1], imgs1)
            # ave_recon_loss_z = (recon_loss_z0 + recon_loss_z1) / 2
            ave_recon_loss_z = torch.zeros(1, device=config['device'])

            loss = config['lambda_recon_z'] * ave_recon_loss_z + \
                   config['lambda_triplet'] * triplet_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config['lr_scheduler'] and e > config['lr_scheduler_warmup_steps']:
                scheduler.step()

            loss_dict['recon_loss_z'] += ave_recon_loss_z.item() if config['lambda_recon_z'] > 0. else 0.
            # loss_dict['recon_loss_proto'] += ave_recon_loss_proto.item() if config['lambda_recon_proto'] > 0. else 0.
            # loss_dict['vq_loss'] += vq_loss.item()
            loss_dict['triplet_loss'] += triplet_loss.item() if config['lambda_pair'] > 0. else 0.
            loss_dict['loss'] += loss.item()

        for key in loss_dict.keys():
            loss_dict[key] /= len(data_loader)

        rtpt.step(subtitle=f'loss={loss_dict["loss"]:2.2f}')

        if (e + 1) % config['display_step'] == 0 or e == config['epochs'] - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", cur_lr, global_step=e)
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

            print(f'SAVED - epoch {e} - imgs @ {config["img_dir"]} - model @ {config["model_dir"]}')


def main(config):

    # get train data
    _data_loader = data.get_dataloader(config)

    batch = next(iter(_data_loader))

    # get test set samples
    test_set = data.get_test_set(_data_loader, config)

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
    _model = GProtoAETriplet(num_hiddens=32, num_residual_layers=2, num_residual_hiddens=32,
                   num_groups=config['n_groups'], num_protos=config['n_protos'],
                   commitment_cost=config['lambda_commitment_cost'], agg_type=config['agg_type'],
                   device=config['device'])

    # checkpoint = torch.load('VQ_VAE/pretrain/gproto_ae_unsup_pretrain.pth', map_location=torch.device('cpu'))
    # _model.load_state_dict(checkpoint['model'])

    _model = _model.to(config['device'])

    # optimizer setup
    optimizer = torch.optim.Adam(_model.parameters(), lr=config['learning_rate'])

    # learning rate scheduler
    scheduler = None
    if config['lr_scheduler']:
        # TODO: try LambdaLR
        num_steps = len(_data_loader) * config['epochs']
        num_steps += config['lr_scheduler_warmup_steps']
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=2e-5)

    # start training
    train(_model, _data_loader, test_set, optimizer, scheduler, writer, config)


if __name__ == '__main__':
    # get config
    config = parse_args_as_dict(sys.argv[1:])

    main(config)
