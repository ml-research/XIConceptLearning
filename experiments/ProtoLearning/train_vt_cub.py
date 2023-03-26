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

import experiments.ProtoLearning.utils as utils
import experiments.ProtoLearning.data as data
from experiments.ProtoLearning.models.vt_cub import VT
from experiments.ProtoLearning.args import parse_args_as_dict
from pytorch_lightning.loggers import WandbLogger


def train(model, data_loader, log_samples, optimizer, scheduler, writer, cur_epoch, config):

    rtpt = RTPT(name_initials=config['initials'], experiment_name='XIC_PrototypeDL', max_iterations=config['epochs'])
    rtpt.start()

    os.environ["WANDB_API_KEY"] =  "your_key_here"

    logger = WandbLogger(name=config['exp_name'], project= "XIConceptLearning", log_model= False)  if config['wandb'] else None

    warmup_steps = cur_epoch * len(data_loader)

    for e in range(cur_epoch, config['epochs']):
        
        max_iter = len(data_loader)
        start = time.time()
        loss_dict = dict(
            {'loss': 0, "proto_recon_loss": 0})

        torch.autograd.set_detect_anomaly(True)
        for i, batch in enumerate(data_loader):

            # manual lr warmup
            if warmup_steps < config['lr_scheduler_warmup_steps']:
                learning_rate = config['learning_rate'] * (warmup_steps + 1) / config['lr_scheduler_warmup_steps']
                optimizer.param_groups[0]['lr'] = learning_rate
            warmup_steps += 1

            imgs, labels_one_hot, shared_labels = batch

            imgs0 = imgs[0].to(config['device'])
            imgs1 = imgs[1].to(config['device'])
            imgs = (imgs0, imgs1)
            shared_labels = shared_labels.to(config['device'])


            # TODO: hack for now
            if e >= 7000:
                model.softmax_temp = .000001
            elif e >= 6000:
                model.softmax_temp = .00001
            elif e >= 5000:
                model.softmax_temp = .0001
            elif e >= 4000:
                model.softmax_temp = .001
            elif e >= 3000:
                model.softmax_temp = .01
            elif e >= 2000:
                model.softmax_temp = .1
            elif e >= 1000:
                model.softmax_temp = .5
            elif e < 1000:
                model.softmax_temp = 2.

            preds, proto_recons = model.forward(imgs, shared_labels)

            # reconstruction loss
            recon_loss_z0_swap_proto = F.mse_loss(proto_recons[2], imgs0)
            recon_loss_z1_swap_proto = F.mse_loss(proto_recons[3], imgs1)
            ave_recon_loss_proto = (recon_loss_z0_swap_proto + recon_loss_z1_swap_proto) / 2

            loss = config['lambda_recon_proto'] * ave_recon_loss_proto

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config['lr_scheduler'] and warmup_steps > config['lr_scheduler_warmup_steps']:
                scheduler.step()

            loss_dict['proto_recon_loss'] += ave_recon_loss_proto.item() if config['lambda_recon_proto'] > 0. else 0.
            loss_dict['loss'] += loss.item()

        for key in loss_dict.keys():
            loss_dict[key] /= len(data_loader)

        rtpt.step(subtitle=f'loss={loss_dict["loss"]:2.2f}')

        if (e + 1) % config['display_step'] == 0 or e == config['epochs'] - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", cur_lr, global_step=e)
            if config['wandb']:
                _log(logger, name="lr", value=cur_lr, epoch=e)
            for key in loss_dict.keys():
                writer.add_scalar(f'train/{key}', loss_dict[key], global_step=e)
                if config['wandb']:
                    _log(logger, name=key, value=loss_dict[key], epoch=e)

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
            torch.save(state, os.path.join(config['model_dir'], '%05d.pth' % (e)))

            if config['extra_mlp_dim'] == 0.:
                utils.plot_prototypes(model, writer, logger, config, step=e)

            # plot a few samples with proto recon
            utils.plot_test_examples(log_samples, model, writer, config, logger=logger, step=e)

            print(f'SAVED - epoch {e} - imgs @ {config["img_dir"]} - model @ {config["model_dir"]}')


def _log(logger, name, value, epoch):
    try:
        logger.experiment.log(
            {name: value, "epoch": epoch})
    except Exception as e:
        print(e)
        print("Something went wrong while trying to log.")
            
def test(model, log_samples, writer, config):
    utils.plot_test_examples(log_samples, model, writer, config, step=0)


def load_pretrained_ae_state_dict(_model, state_dict):

    model_state = _model.state_dict()
    for name, param in state_dict.items():
        if "_encoder" in name:
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            model_state[name].copy_(param)
    _model.load_state_dict(model_state)

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
    # hard set to make it same as pretrained ae model
    assert config['lin_enc_size'] == 512

    # create a list of start and stop id for each prototype group
    config['prototype_cumsum'] = list(np.cumsum(config['prototype_vectors']))
    config['prototype_cumsum'].insert(0, 0)

    if config['dataset'] != 'cub':
        print(f"Dataset is '{config['dataset']}'. You might want to change it to 'cub'.")
    # get train data
    _data_loader = data.get_dataloader(config)

    # get test set samples
    test_set = data.get_test_set(_data_loader, config)

    # create tb writer
    writer = SummaryWriter(log_dir=config['results_dir'])

    # model setup
    _model = VT(num_hiddens=50, num_residual_layers=2, num_residual_hiddens=50,
                    n_proto_vecs=config['prototype_vectors'], lin_enc_size=config['lin_enc_size'],
                    proto_dim=config['proto_dim'], softmax_temp=config['temperature'],
                    extra_mlp_dim=config['extra_mlp_dim'],
                    multiheads=config['multiheads'], train_protos=config['train_protos'],
                    device=config['device'])

    _model = _model.to(config['device'])

    # train swap disentanglement from scratch
    if config['ckpt_fp'] is None:
        cur_epoch = 0
    # load ckpt file from which to continue training
    else:
        ckpt = torch.load(config['ckpt_fp'], map_location=torch.device(config['device']))
        cur_epoch = ckpt['ep']
        _model.load_state_dict(ckpt['model'])
        _model.proto_dict = ckpt['model_misc']['prototypes']
        _model.softmax_temp = ckpt['model_misc']['softmax_temp']
        print(f"loading {config['ckpt_fp']} from epoch {cur_epoch} for further training")

    # optimizer setup
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, _model.parameters()), lr=config['learning_rate'])

    # learning rate scheduler
    scheduler = None
    if config['lr_scheduler']:
        num_steps = len(_data_loader) * config['epochs'] - cur_epoch
        num_steps += config['lr_scheduler_warmup_steps']
    if not config['test']:
        # start training
        train(_model, _data_loader, test_set, optimizer, scheduler, writer, cur_epoch, config)
    else:
        test(_model, test_set, writer, config)

if __name__ == '__main__':
    # get config
    config = parse_args_as_dict(sys.argv[1:])

    main(config)
