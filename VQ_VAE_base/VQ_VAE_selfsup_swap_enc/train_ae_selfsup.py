import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib

matplotlib.use('Agg')
import sys
sys.path.append("Proto_Cat_VAE/models/")
import os
from torch.utils.tensorboard import SummaryWriter
from rtpt.rtpt import RTPT
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter

from torch.optim import Adam

import VQ_VAE_selfsup_swap_enc.utils as utils
import VQ_VAE_selfsup_swap_enc.data as data
from VQ_VAE_selfsup_swap_enc.models.ae_selfsup import AE_selfsup
from VQ_VAE_selfsup_swap_enc.args import parse_args_as_dict


def train(model, data_loader, log_samples, optimizer, scheduler, writer, config):

    rtpt = RTPT(name_initials=config['initials'], experiment_name='XIC_PrototypeDL', max_iterations=config['epochs'])
    rtpt.start()

    for e in range(0, config['epochs']):
        max_iter = len(data_loader)
        start = time.time()
        loss_dict = dict(
            {'recon_loss_z': 0, 'loss': 0, 'triplet_loss': 0})

        torch.autograd.set_detect_anomaly(True)
        for i, batch in enumerate(data_loader):
            imgs0, imgs0_a, imgs1 = batch[0]
            imgs0 = imgs0.to(config['device'])
            imgs0_a = imgs0_a.to(config['device'])
            imgs1 = imgs1.to(config['device'])
            imgs = (imgs0, imgs0_a, imgs1)

            # labels = batch[1]
            # labels = labels.to(config['device'])

            imgs_recon, triplet_loss, distances, distances_emb = model.forward(imgs, k=None)

            # plot training images
            if i == 0 and (((e + 1) % config['save_step'] == 0) or e == config['epochs'] - 1 or e == 0):
                utils.plot_train_examples_2(imgs, imgs_recon, distances, distances_emb, writer, config, batch=i, step=e)
                utils.plot_train_examples_sanity(model, imgs, writer, config, batch=i, step=e)

            # # reconstruciton loss for z
            recon_loss_z0 = F.mse_loss(imgs_recon[0], imgs0)
            recon_loss_z0_a = F.mse_loss(imgs_recon[1], imgs0_a)
            recon_loss_z1 = F.mse_loss(imgs_recon[2], imgs1)
            ave_recon_loss_z = (recon_loss_z0 + recon_loss_z0_a + recon_loss_z1) / 3

            loss = config['lambda_recon_z'] * ave_recon_loss_z + \
                   config['lambda_triplet'] * triplet_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config['lr_scheduler'] and e > config['lr_scheduler_warmup_steps']:
                scheduler.step()

            loss_dict['recon_loss_z'] += ave_recon_loss_z.item() if config['lambda_recon_z'] > 0. else 0.
            loss_dict['triplet_loss'] += triplet_loss.item() if config['lambda_triplet'] > 0. else 0.
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

            # # plot a few samples with proto recon
            # utils.plot_examples(log_samples, model, writer, config, step=e)

            print(f'SAVED - epoch {e} - imgs @ {config["img_dir"]} - model @ {config["model_dir"]}')


def load_pretrained_state_dict(_model, state_dict):

    model_state = _model.state_dict()
    for name, param in state_dict.items():
        # if name not in model_state:
        #     continue
        if "_encoder" not in name:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        model_state[name].copy_(param)


def freeze_encoder(model):
    """
    Freezes _encoder and _encoder_linear
    :param model:
    :return:
    """
    for name, p in model.named_parameters():
        if "_encoder" in name:
            p.requires_grad = False


def main(config):

    # get train data
    _data_loader = data.get_dataloader(config)

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
    _model = AE_selfsup(num_hiddens=64, num_residual_layers=2, num_residual_hiddens=64,
                        enc_size=(config['n_groups']+1) * 256, num_groups=config['n_groups'],
                        num_protos=config['n_protos'], agg_type='linear', device=config['device'])

    _model = _model.to(config['device'])

    # load pretrained AE
    ckpt = torch.load('VQ_VAE_take2/pretrained/00049.pth', map_location=torch.device(config['device']))
    load_pretrained_state_dict(_model, ckpt['model'])

    # freeze encoder and encoder linear
    freeze_encoder(_model)

    # optimizer setup
    optimizer = torch.optim.Adam(_model.parameters(), lr=config['learning_rate'])

    # learning rate scheduler
    scheduler = None
    if config['lr_scheduler']:
        num_steps = len(_data_loader) * config['epochs']
        num_steps += config['lr_scheduler_warmup_steps']
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=2e-5)

    # start training
    train(_model, _data_loader, test_set, optimizer, scheduler, writer, config)


if __name__ == '__main__':
    # get config
    config = parse_args_as_dict(sys.argv[1:])

    main(config)
