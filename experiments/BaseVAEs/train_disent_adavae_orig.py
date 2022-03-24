import torch
import numpy as np
import time
import matplotlib

matplotlib.use('Agg')
import sys
sys.path.append("experiments/BaseVAEs/models/")
import os
from torch.utils.tensorboard import SummaryWriter
from rtpt.rtpt import RTPT
from torch.optim import lr_scheduler

from torch.optim import Adam
from disent.model.ae import EncoderConv64, DecoderConv64, AutoEncoder
from disent.frameworks.vae.weaklysupervised import AdaVae

import experiments.BaseVAEs.utils_disent as utils
import experiments.BaseVAEs.data as data
from experiments.BaseVAEs.args import parse_args_as_dict


def train(model, data_loader, log_samples, optimizer, scheduler, writer, config):

    rtpt = RTPT(name_initials=config['initials'], experiment_name='XIC_PrototypeDL', max_iterations=config['epochs'])
    rtpt.start()

    warmup_steps = 0

    for e in range(0, config['epochs']):
        max_iter = len(data_loader)
        start = time.time()
        loss_dict = dict(
            {'z_recon_loss': 0, 'loss': 0, 'kld': 0, 'elbo': 0})

        for i, batch in enumerate(data_loader):

            # manual lr warmup
            if warmup_steps < config['lr_scheduler_warmup_steps']:
                learning_rate = config['learning_rate'] * (warmup_steps + 1) / config['lr_scheduler_warmup_steps']
                optimizer.param_groups[0]['lr'] = learning_rate
            warmup_steps += 1

            imgs, labels_one_hot, labels_id, shared_labels = batch

            imgs0 = imgs[0].to(config['device'])
            imgs1 = imgs[1].to(config['device'])
            imgs = (imgs0, imgs1)
            # labels0_one_hot = labels_one_hot[0].to(config['device']).float()
            # labels1_one_hot = labels_one_hot[1].to(config['device']).float()
            # labels0_ids = labels_id[0].to(config['device']).float()
            # labels1_ids = labels_id[1].to(config['device']).float()
            shared_labels = None

            # from disent repo: x_targ is if augmentation is applied, otherwise x_targ is x
            batch = {'x': imgs, 'x_targ': imgs, 'shared_mask': shared_labels}
            batch_loss_dict = model.compute_training_loss(batch, batch_idx=i)

            loss, recon_loss, kl_reg_loss, kl_loss, elbo = batch_loss_dict['train_loss'], \
                                                           batch_loss_dict['recon_loss'], \
                                                           batch_loss_dict['kl_reg_loss'], \
                                                           batch_loss_dict['kl_loss'], \
                                                           batch_loss_dict['elbo']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config['lr_scheduler'] and e > config['lr_scheduler_warmup_steps']:
                scheduler.step()

            loss_dict['z_recon_loss'] += recon_loss.item()
            # loss_dict['proto_recon_loss'] += proto_recon_loss.item()
            loss_dict['kld'] += kl_reg_loss.item()
            loss_dict['loss'] += loss.item()
            loss_dict['elbo'] += elbo.item()

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

            # plot a few samples with proto recon
            utils.plot_examples(log_samples, model, writer, config, step=e)

            print(f'SAVED - epoch {e} - imgs @ {config["img_dir"]} - model @ {config["model_dir"]}')


def main(config):

    # get train data
    _data_loader = data.get_dataloader(config)

    # get test set samples
    test_set = data.get_test_set(_data_loader, config)

    # create tb writer
    writer = SummaryWriter(log_dir=config['results_dir'])

    # model setup
    _model = AdaVae(make_optimizer_fn=lambda params: Adam(params, lr=1e-3),
                 make_model_fn=lambda: AutoEncoder(
                     encoder=EncoderConv64(x_shape=(3, 64, 64), z_size=config['n_groups'], z_multiplier=2),
                     decoder=DecoderConv64(x_shape=(3, 64, 64), z_size=config['n_groups']),
                 ),
                 cfg=AdaVae.cfg(beta=config['beta'], average_mode='gvae', symmetric_kl=False))

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
