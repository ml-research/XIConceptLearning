import torch
import numpy as np
import time
import matplotlib

matplotlib.use('Agg')
import sys
import os
from args import parse_args_as_dict
from torch.utils.tensorboard import SummaryWriter
from rtpt.rtpt import RTPT
from torch.optim import lr_scheduler

import pair_prototype_nongroup_to_group.utils as utils
import pair_prototype_nongroup_to_group.losses as losses
import pair_prototype_nongroup_to_group.data as data
from pair_prototype_nongroup_to_group.model import Pair_RAE


def train(model, data_loader, log_samples, optimizer, scheduler, writer, config):

    rtpt = RTPT(name_initials=config['initials'], experiment_name='XIC_PrototypeDL', max_iterations=config['epochs'])
    rtpt.start()

    mse = torch.nn.MSELoss()

    for e in range(0, config['epochs']):
        max_iter = len(data_loader)
        start = time.time()
        loss_dict = dict(
            {'img_recon_loss': 0, 'softmin_proto_recon_loss': 0, 'r1_loss': 0, 'r2_loss': 0,
             'r5_loss': 0, 'pair_loss': 0, 'loss': 0, 'enc_mse_loss': 0, 'ad_loss': 0})

        for i, batch in enumerate(data_loader):
            imgs1, imgs2 = batch[0]
            imgs1 = imgs1.to(config['device'])
            imgs2 = imgs2.to(config['device'])
            imgs = torch.cat((imgs1, imgs2), dim=0)
            # labels1, labels2 = batch[1]

            std = (config['epochs'] - e) / config['epochs']

            res_dict = model.forward((imgs1, imgs2), std)

            rec_imgs, rec_protos, dists, s_weights, feature_vecs_z, \
            proto_vecs, agg_protos, pair_s_weights = utils.unfold_res_dict(res_dict)

            # enforces the same prototype to be chosen for one group between a pair of imgs, i.e. one prototype should
            # be the same for both imgs
            pair_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_pair'] != 0:
                pair_loss = losses.pair_loss(pair_s_weights)

            # draws prototype close to training example
            r1_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_r1'] != 0:
                r1_loss = losses.r1_loss(proto_vecs, feature_vecs_z, model.dim_proto, config)

            # draws encoding close to prototype
            r2_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_r2'] != 0:
                r2_loss = losses.r2_loss(proto_vecs, feature_vecs_z, model.dim_proto, config)

            loss_ad = torch.zeros((1,)).to(config['device'])
            if config['lambda_ad'] != 0:
            # for k in range(len(proto_vecs)):
            #     loss_ad += torch.mean(torch.sqrt(torch.sum(proto_vecs[k].T ** 2, dim=1)), dim=0)
                loss_ad = losses.ad_loss(proto_vecs)

            softmin_proto_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_softmin_proto'] != 0:
                softmin_proto_recon_loss = mse(rec_protos, imgs)

            img_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_z'] != 0:
                img_recon_loss = mse(rec_imgs, imgs)

            img_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_enc_mse'] != 0:
                loss_enc_mse = mse(agg_protos, feature_vecs_z.flatten(1, 3))

            loss = config['lambda_z'] * img_recon_loss + \
                   config['lambda_softmin_proto'] * softmin_proto_recon_loss + \
                   config['lambda_r1'] * r1_loss + \
                   config['lambda_r2'] * r2_loss + \
                   config['lambda_enc_mse'] * loss_enc_mse + \
                   config['lambda_ad'] * loss_ad + \
                   config['lambda_pair'] * pair_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config['lr_scheduler'] and e > config['lr_scheduler_warmup_steps']:
                scheduler.step()

            loss_dict['img_recon_loss'] += img_recon_loss.item()
            loss_dict['softmin_proto_recon_loss'] += softmin_proto_recon_loss.item()

            loss_dict['r1_loss'] += r1_loss.item()
            loss_dict['r2_loss'] += r2_loss.item()
            loss_dict['enc_mse_loss'] += loss_enc_mse.item()
            loss_dict['ad_loss'] += loss_ad.item()
            loss_dict['pair_loss'] += pair_loss.item()

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

            # plot the indivisual prototypes of each group
            utils.plot_prototypes(model, proto_vecs, writer, config, step=e)

            # plot a few samples with proto recon
            utils.plot_examples(log_samples, model, writer, config, step=e)

            print(f'SAVED - epoch {e} - imgs @ {config["img_dir"]} - model @ {config["model_dir"]}')


def main(config):

    assert config['dataset'] == 'toycolorshapepairs'

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
    _model = Pair_RAE(input_dim=(1, config['img_shape'][0], config['img_shape'][1], config['img_shape'][2]),
                 n_z=config['n_z'], filter_dim=config['filter_dim'],
                 n_proto_vecs=config['prototype_vectors'],
                 train_pw=config['train_weighted_protos'],
                 device=config['device'],
                 agg_type=config['agg_type'])

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