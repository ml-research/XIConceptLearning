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

from torch.optim import Adam

import VQ_VAE_selfsup_swap_enc.utils as utils
import VQ_VAE_selfsup_swap_enc.data as data
from VQ_VAE_selfsup_swap_enc.models.ae import AE
from VQ_VAE_selfsup_swap_enc.args import parse_args_as_dict


def train(model, data_loader, log_samples, optimizer, scheduler, writer, config):

    rtpt = RTPT(name_initials=config['initials'], experiment_name='XIC_PrototypeDL', max_iterations=config['epochs'])
    rtpt.start()

    for e in range(0, config['epochs']):
        max_iter = len(data_loader)
        start = time.time()
        loss_dict = dict(
            {'loss': 0})

        torch.autograd.set_detect_anomaly(True)
        for i, batch in enumerate(data_loader):
            imgs0, imgs0_a, imgs1 = batch[0]
            imgs0 = imgs0.to(config['device'])
            imgs0_a = imgs0_a.to(config['device'])
            imgs1 = imgs1.to(config['device'])
            imgs = (imgs0, imgs0_a, imgs1)

            # labels = batch[1]
            # labels = labels.to(config['device'])

            imgs_recon = model(imgs)

            # plot training images
            if i == 0 and (((e + 1) % config['save_step'] == 0) or e == config['epochs'] - 1 or e == 0):
                utils.plot_train_examples(imgs, imgs_recon, writer, config, step=e)

            # # reconstruciton loss for z
            recon_loss_z0 = F.mse_loss(imgs_recon[0], imgs0)
            recon_loss_z0_a = F.mse_loss(imgs_recon[1], imgs0_a)
            recon_loss_z1 = F.mse_loss(imgs_recon[2], imgs1)
            loss = (recon_loss_z0 + recon_loss_z0_a + recon_loss_z1) / 3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config['lr_scheduler'] and e > config['lr_scheduler_warmup_steps']:
                scheduler.step()

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
    _model = AE(num_hiddens=64, num_residual_layers=2, num_residual_hiddens=64, enc_size=(config['n_groups']+1) * 256)

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
