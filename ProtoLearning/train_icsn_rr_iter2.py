import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib

matplotlib.use('Agg')
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from rtpt.rtpt import RTPT
from torch.nn.parameter import Parameter

import ProtoLearning.utils as utils
import ProtoLearning.data as data
from ProtoLearning.models.icsn import iCSN
from ProtoLearning.args import parse_args_as_dict


def train(model, data_loader, log_samples, optimizer, scheduler, writer, cur_epoch, config):

    rtpt = RTPT(name_initials=config['initials'], experiment_name='XIC_PrototypeDL', max_iterations=config['epochs'])
    rtpt.start()

    warmup_steps = cur_epoch * len(data_loader)

    for e in range(cur_epoch, config['epochs']):
        max_iter = len(data_loader)
        start = time.time()
        loss_dict = dict(
            {'loss': 0, "proto_recon_loss": 0, "rr_loss": 0})

        print(model.softmax_temp)

        torch.autograd.set_detect_anomaly(True)
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
            labels0_ids = labels_id[0].to(config['device']).float()
            labels1_ids = labels_id[1].to(config['device']).float()
            shared_labels = shared_labels.to(config['device'])

            # set the correct prototype slot id for pentagons and circles
            if config['seed'] == 0:
                pent_one_hot = torch.tensor([1., 0., 0., 0., 0., 0.], device=config['device'])
                circle_one_hot = torch.tensor([0., 0., 0., 0., 0., 1.], device=config['device'])
            elif config['seed'] == 13:
                pent_one_hot = torch.tensor([0., 0., 0., 0., 1., 0.], device=config['device'])
                circle_one_hot = torch.tensor([1., 0., 0., 0., 0., 0.], device=config['device'])

            if e <= 200:
                model.softmax_temp = .0001

            preds, proto_recons = model.forward(imgs, shared_labels)

            # get ids of samples that contain a pentagon
            pent_ids0 = torch.where(labels0_ids[:, 1] == 3)[0]
            pent_ids1 = torch.where(labels1_ids[:, 1] == 3)[0]
            # get ids of samples that contain a circle
            circle_ids0 = torch.where(labels0_ids[:, 1] == 1)[0]
            circle_ids1 = torch.where(labels1_ids[:, 1] == 1)[0]

            # for those samples that correspond to a pentagon state that the prototype should be a different one than
            # that currently predicted, but make sure those that correspond to a circle stick predicted prototype
            if len(pent_ids0) > 0:
                rr_loss_0_0 = F.mse_loss(preds[0][pent_ids0, 6:12],
                                         pent_one_hot.repeat((pent_ids0.shape[0], 1)))
            else:
                rr_loss_0_0 = torch.zeros(1, device=config['device'])
            if len(circle_ids0) > 0:
                rr_loss_0_1 = F.mse_loss(preds[0][circle_ids0, 6:12],
                                         circle_one_hot.repeat((circle_ids0.shape[0], 1)))
            else:
                rr_loss_0_1 = torch.zeros(1, device=config['device'])

            if len(pent_ids1) > 0:
                rr_loss_1_0 = F.mse_loss(preds[1][pent_ids1, 6:12],
                                         pent_one_hot.repeat((pent_ids1.shape[0], 1)))
            else:
                rr_loss_1_0 = torch.zeros(1, device=config['device'])
            if len(circle_ids1) > 0:
                rr_loss_1_1 = F.mse_loss(preds[1][circle_ids1, 6:12],
                                         circle_one_hot.repeat((circle_ids1.shape[0], 1)))
            else:
                rr_loss_1_1 = torch.zeros(1, device=config['device'])

            ave_rr_loss = ((rr_loss_0_0 + rr_loss_0_1) + (rr_loss_1_0 + rr_loss_1_1))/2

            wr_loss_0 = F.mse_loss(preds[0][:, config['wrong_protos']], torch.zeros((preds[0].shape[0],
                                                                                    len(config['wrong_protos'])),
                                                                                    device=config['device']))
            wr_loss_1 = F.mse_loss(preds[1][:, config['wrong_protos']], torch.zeros((preds[0].shape[0],
                                                                                    len(config['wrong_protos'])),
                                                                                    device=config['device']))
            ave_wr_loss = (wr_loss_0 + wr_loss_1)/2


            # reconstruciton loss
            # recon_loss_z0_proto = F.mse_loss(proto_recons[0], imgs0)
            # recon_loss_z1_proto = F.mse_loss(proto_recons[1], imgs1)
            recon_loss_z0_swap_proto = F.mse_loss(proto_recons[2], imgs0)
            recon_loss_z1_swap_proto = F.mse_loss(proto_recons[3], imgs1)
            ave_recon_loss_proto = (recon_loss_z0_swap_proto + recon_loss_z1_swap_proto) / 2

            loss = config['lambda_recon_proto'] * ave_recon_loss_proto + \
                   config['lambda_rr'] * (ave_rr_loss + ave_wr_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config['lr_scheduler'] and warmup_steps > config['lr_scheduler_warmup_steps']:
                scheduler.step()

            loss_dict['proto_recon_loss'] += ave_recon_loss_proto.item() if config['lambda_recon_proto'] > 0. else 0.
            loss_dict['rr_loss'] += ave_rr_loss.item() if config['lambda_rr'] > 0. else 0.
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
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f'epoch {e} - loss {loss.item():2.4f} - time/epoch {(time.time() - start):2.2f} - lr {cur_lr} '
                  f'- softmax_temp {model.softmax_temp}')
            loss_summary = ''
            for key in loss_dict.keys():
                loss_summary += f'{key} {loss_dict[key]:2.4f} '
            print(loss_summary)

        if (e + 1) % config['save_step'] == 0 or e == config['epochs'] - 1 or e == 0:
            state = {
                'model': model.state_dict(),
                'model_misc': {'prototypes': model.proto_dict,
                               'softmax_temp': model.softmax_temp},
                'optimizer': optimizer.state_dict(),
                'ep': e,
                'config': config
            }
            # torch.save(state, os.path.join(writer.log_dir, f"{config['exp_name']}.pth"))
            torch.save(state, os.path.join(config['model_dir'], '%05d.pth' % (e)))

            # TODO: how to plot the prorotypes with extra encoding?
            if config['extra_mlp_dim'] == 0.:
                utils.plot_prototypes(model, writer, config, step=e)

            # plot a few samples with proto recon
            utils.plot_test_examples(log_samples, model, writer, config, step=e)

            print(f'SAVED - epoch {e} - imgs @ {config["img_dir"]} - model @ {config["model_dir"]}')


def test(model, log_samples, writer, config):
    utils.plot_test_examples(log_samples, model, writer, config, step=0)


def main(config):
    # hard set to make it same as pretrained ae model
    assert config['lin_enc_size'] == 512

    # create a list of start and stop id for each prototype group
    config['prototype_cumsum'] = list(np.cumsum(config['prototype_vectors']))
    config['prototype_cumsum'].insert(0, 0)

    # get train data
    _data_loader = data.get_dataloader(config)

    # get test set samples
    test_set = data.get_test_set(_data_loader, config)

    # create tb writer
    writer = SummaryWriter(log_dir=config['results_dir'])

    # model setup
    _model = iCSN(num_hiddens=64, num_residual_layers=2, num_residual_hiddens=64,
                    n_proto_vecs=config['prototype_vectors'], lin_enc_size=config['lin_enc_size'],
                    proto_dim=config['proto_dim'], softmax_temp=config['temperature'],
                    extra_mlp_dim=config['extra_mlp_dim'],
                    multiheads=config['multiheads'], train_protos=config['train_protos'],
                    device=config['device'])

    _model = _model.to(config['device'])

    cur_epoch = 0
    # load ckpt file from which to continue training
    if config['ckpt_fp'] is not None:
        print(f"loading {config['ckpt_fp']} for further training")
        ckpt = torch.load(config['ckpt_fp'], map_location=torch.device(config['device']))
        # cur_epoch = ckpt['ep']
        _model.load_state_dict(ckpt['model'])
        _model.proto_dict = ckpt['model_misc']['prototypes']
        _model.softmax_temp = ckpt['model_misc']['softmax_temp']
        # _model.softmax_temp = config['temperature']

    # optimizer setup
    optimizer = torch.optim.Adam(_model.parameters(), lr=config['learning_rate'])
    # optimizer = torch.optim.SGD(_model.parameters(), lr=config['learning_rate'], momentum=0.9)

    # learning rate scheduler
    scheduler = None
    if config['lr_scheduler']:
        num_steps = len(_data_loader) * config['epochs'] - cur_epoch
        num_steps += config['lr_scheduler_warmup_steps']
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=2e-5)
        # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.0004, step_size_up=100)
    if not config['test']:
        # start training
        train(_model, _data_loader, test_set, optimizer, scheduler, writer, cur_epoch, config)
    else:
        test(_model, test_set, writer, config)

if __name__ == '__main__':
    # get config
    config = parse_args_as_dict(sys.argv[1:])

    main(config)
