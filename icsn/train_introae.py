import torch
import torch.nn.functional as F
import time
import matplotlib

matplotlib.use('Agg')
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from rtpt.rtpt import RTPT
from torch.nn.parameter import Parameter
from tqdm import tqdm

import icsn.utils as utils
import icsn.data as data
from icsn.models.introae import IntroAE
from icsn.args import parse_args_as_dict


def train(model, data_loader, test_loader, optimizer, writer, cur_epoch, config):

    rtpt = RTPT(name_initials=config['initials'], experiment_name='XIC_PrototypeDL', max_iterations=config['epochs'])
    rtpt.start()

    warmup_steps = cur_epoch * len(data_loader)

    print("Begin training!")

    for e in range(cur_epoch, config['epochs']):
        max_iter = len(data_loader)
        start = time.time()
        loss_dict = dict(
            {'loss': 0})

        torch.autograd.set_detect_anomaly(True)
        for i, batch in tqdm(enumerate(data_loader)):
            # manual lr warmup
            if warmup_steps < config['lr_scheduler_warmup_steps']:
                learning_rate = config['learning_rate'] * (warmup_steps + 1) / config['lr_scheduler_warmup_steps']
                optimizer.param_groups[0]['lr'] = learning_rate
            warmup_steps += 1

            imgs, _, _, _ = batch

            imgs0 = imgs[0].to(config['device'])
            imgs1 = imgs[1].to(config['device'])

            imgs = torch.cat((imgs0, imgs1), dim=0)

            _, recons = model.forward(imgs)

            # reconstruciton loss
            loss = F.mse_loss(recons, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            print(f'epoch {e} - loss {loss.item():2.4f} - time/epoch {(time.time() - start):2.2f} - lr {cur_lr} ')
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
            # torch.save(state, os.path.join(writer.log_dir, f"{config['exp_name']}.pth"))
            torch.save(state, os.path.join(config['model_dir'], '%05d.pth' % (e)))

            # plot a few samples with recon
            imgs = next(iter(test_loader))
            imgs = imgs.to(config['device'])
            _, recons = model.forward(imgs)
            utils.plot_examples(imgs, recons, writer, config, step=e)

            print(f'SAVED - epoch {e} - imgs @ {config["img_dir"]} - model @ {config["model_dir"]}')


def test(model, test_loader, writer, config):
    # plot a few samples with recon
    imgs = next(iter(test_loader))
    imgs = imgs.to(config['device'])
    preds, recons = model.forward_single({'imgs': imgs})
    utils.plot_examples_code(imgs, recons,
                             preds.reshape(preds.shape[0], model.n_groups, -1),
                             writer, config, step=0)


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

    # get train data
    _data_loader = data.get_dataloader(config)

    # get test set samples
    test_loader = data.get_test_set(_data_loader, config)

    # create tb writer
    writer = SummaryWriter(log_dir=config['results_dir'])

    # model setup
    _model = IntroAE(cdim=3, hdim=512, channels=[64, 128, 256, 512], image_size=64)

    _model = _model.to(config['device'])

    cur_epoch = 0

    # optimizer setup
    optimizer = torch.optim.Adam(_model.parameters(), lr=config['learning_rate'])

    if not config['test']:
        # start training
        train(_model, _data_loader, test_loader, optimizer, writer, cur_epoch, config)
    else:
        test(_model, test_loader, writer, config)

if __name__ == '__main__':
    # get config
    config = parse_args_as_dict(sys.argv[1:])

    main(config)
