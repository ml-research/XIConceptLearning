import torch
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import os
import datetime
import json
from torch.utils.tensorboard import SummaryWriter
from rtpt.rtpt import RTPT
from torch.optim import lr_scheduler

import utils as utils
import losses as losses
import data as data
from model import RAE

# TODO: convert config fully to argparse?
config = dict({
    # TODO: for debugging
    # 'device': 'cuda',
    'device': 'cpu',

    'save_step': 50,
    'print_step': 10,
    'display_step': 1,

    'mse': True,  # use MSE instead of mean(list_of_norms) for reconstruction loss

    'lambda_min_proto': 0,  # decode protoype with min distance to z
    'lambda_z': 0,  # decode z
    'lambda_softmin_proto': 5,  # decode softmin weighted combination of prototypes
    'lambda_r1': 1e-2,  # draws prototype close to training example
    'lambda_r2': 0, #1e-2,  # draws encoding close to prototype
    'lambda_enc_mse': 0.05, #0.5 for linear, conv; 0.05 for attention
    'lambda_ad': 0,#1e-2,
    'lambda_pair': 1,#1e-2,
    'diversity_threshold': 2,  # 1-2 suggested by paper
    'train_weighted_protos': False,

    'learning_rate': 1e-3,
    'lr_scheduler': False,
    'lr_scheduler_warmup_steps': 800,
    # TODO: for debugging
    # 'batch_size': 1000,
    'batch_size': 7,
    'n_workers': 2,
    'n_prototype_vectors': [4, 2],
    'filter_dim': 32,
    'n_z': 10,
    'batch_elastic_transform': False,
    'sigma': 4,
    'alpha': 20,

    'experiment_name': '',
    'results_dir': 'results_group',
    'model_dir': 'states',
    'img_dir': 'imgs',

    'dataset': 'toycolorpairs',  # 'toycolor' or 'mnist' or 'toycolorshape'
    'init': 'xavier',
})


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--seed", type=int, default=0, help="seed"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=500, help="num of epochs to train"
    )
    parser.add_argument(
        "--agg-type", type=str, default='sum', help="type of prototype aggregation layer"
    )

    parser.add_argument(
        "-pv", "--prototype-vectors", nargs="+", default='4,4',
        help="List of img shape dims [#p1, #p2, ...]"
    )

    args = parser.parse_args()
    args.prototype_vectors = [int(n) for n in args.prototype_vectors[0].split(',')]
    return args

# TODO: divide into separate run_step fct
def train(model, data_loader, log_samples, writer):
    # optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    # learning rate scheduler
    if config['lr_scheduler']:
        # TODO: try LambdaLR
        num_steps = len(data_loader) * config['training_epochs']
        num_steps += config['lr_scheduler_warmup_steps']
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=2e-5)


    rtpt = RTPT(name_initials='MM', experiment_name='XIC_PrototypeDL', max_iterations=config['training_epochs'])
    rtpt.start()

    mse = torch.nn.MSELoss()

    for e in range(0, config['training_epochs']):
        max_iter = len(data_loader)
        start = time.time()
        loss_dict = dict(
            {'img_recon_loss': 0, 'min_proto_recon_loss': 0, 'softmin_proto_recon_loss': 0, 'r1_loss': 0, 'r2_loss': 0,
             'r5_loss': 0, 'pair_loss': 0, 'loss': 0, 'enc_mse_loss': 0, 'ad_loss': 0})

        for i, batch in enumerate(data_loader):
            imgs1, imgs2 = batch[0]
            imgs1 = imgs1.to(config['device'])
            imgs2 = imgs2.to(config['device'])
            imgs = torch.cat((imgs1, imgs2), dim=0)
            # labels1, labels2 = batch[1]

            # TODO: can we just remove the lines to batch_elastic, if we never use it?
            # # to apply elastic transform, batch has to be flattened
            # # store original size to put it back into orginial shape after transformation
            # imgs_shape = imgs.shape
            # if config['batch_elastic_transform']:
            #     imgs = batch_elastic_transform(imgs.view(config['batch_size'], -1), sigma=config['sigma'],
            #                                    alpha=config['alpha'], height=config['img_shape'][1],
            #                                    width=config['img_shape'][2])
            #     imgs = torch.reshape(torch.tensor(imgs), imgs_shape)

            std = (config['training_epochs'] - e) / config['training_epochs']

            res_dict = model.forward((imgs1, imgs2), std)

            rec_imgs, rec_protos, dists, feature_vectors_z, \
            prototype_vectors, mixed_prototypes, pair_s_weights = utils.unfold_res_dict(res_dict)

            # enforces the same prototype to be chosen for one group between a pair of imgs, i.e. one prototype should
            # be the same for both imgs
            pair_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_pair'] != 0:
                pair_loss = losses.pair_loss(pair_s_weights)

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

            # TODO: can we remove this?
            # # diversity regularization
            # # draw prototypes away from each other
            # # Source: Interpretable and Steerable Sequence Learning via Prototypes
            # relu = torch.nn.ReLU()
            # r5_loss = torch.zeros((1,)).to(config['device'])
            # # only compute if needed, cause time you know...
            # if config['lambda_r5'] != 0:
            #     for i in range(prototype_vectors.shape[1]):
            #         for j in range(i + 1, prototype_vectors.shape[1]):
            #             # torch.linalg.norm for torch 1.7.0
            #             max_distance = relu(
            #                 config['diversity_threshold'] - torch.norm(prototype_vectors[i] - prototype_vectors[j]))
            #             r5_loss += max_distance ** 2

            softmin_proto_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_softmin_proto'] != 0:
                softmin_proto_recon_loss = mse(rec_protos, imgs)

            img_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_z'] != 0:
                img_recon_loss = mse(rec_imgs, imgs)

            # TODO: do we need this?
            # get prototype with min distance to z
            min_proto_recon_loss = torch.zeros((1,)).to(config['device'])
            if config['lambda_min_proto'] != 0:
                min_prototype_vectors = prototype_vectors[torch.argmin(dists, 2)] # [batch, n_groups]
                # TODO: we now have to handle that there are one min prototype per group --> maybe move to model?
                # rec_proto = model.dec.forward(min_prototype_vector.reshape(feature_vectors_z.shape))
                # mse = torch.nn.MSELoss()
                # min_proto_recon_loss = mse(rec_proto, imgs)

            loss_enc_mse = mse(mixed_prototypes, feature_vectors_z.flatten(1,3))

            loss = config['lambda_z'] * img_recon_loss + \
                   config['lambda_min_proto'] * min_proto_recon_loss + \
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
            loss_dict['min_proto_recon_loss'] += min_proto_recon_loss.item()
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

        if (e + 1) % config['display_step'] == 0 or e == config['training_epochs'] - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", cur_lr, global_step=e)
            for key in loss_dict.keys():
                writer.add_scalar(f'train/{key}', loss_dict[key], global_step=e)

        if (e + 1) % config['print_step'] == 0 or e == config['training_epochs'] - 1:
            print(f'epoch {e} - loss {loss.item():2.4f} - time/epoch {(time.time() - start):2.2f}')
            loss_summary = ''
            for key in loss_dict.keys():
                loss_summary += f'{key} {loss_dict[key]:2.4f} '
            print(loss_summary)

        if (e + 1) % config['save_step'] == 0 or e == config['training_epochs'] - 1 or e == 0:

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


def main(config):
    # get data
    _data_loader, x_set = data.init_dataset(config)

    # create tb writer
    writer = SummaryWriter(log_dir=config['results_dir'])

    # store config
    print(config['n_prototype_vectors'])
    with open(os.path.join(config['results_dir'], 'args.json'), 'w') as json_file:
        json.dump(config, json_file, indent=2)

    # model setup
    _model = RAE(input_dim=(1, config['img_shape'][0], config['img_shape'][1], config['img_shape'][2]),
                 n_z=config['n_z'], filter_dim=config['filter_dim'],
                 n_prototype_vectors=config['n_prototype_vectors'],
                 train_pw=config['train_weighted_protos'],
                 device=config['device'],
                 agg_type=config['agg_type'])

    _model = _model.to(config['device'])

    # start training
    train(_model, _data_loader, x_set, writer)


if __name__ == '__main__':
    _args = get_args()
    config['seed'] = _args.seed
    # TODO: before push set to old
    # config['n_prototype_vectors'] = _args.prototype_vectors
    config['n_prototype_vectors'] = [4, 2]
    config['training_epochs'] = _args.epochs
    config['agg_type'] = _args.agg_type
    config['n_prototype_groups'] = len(_args.prototype_vectors)
    if config['experiment_name'] == '':
        config['experiment_name'] = 'seed' + str(config['seed']) + '_' \
                                    + 'protos' + str(config['n_prototype_vectors']) + '_' + \
                                    str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]

    config['results_dir'] = os.path.join(config['results_dir'], config['experiment_name'])
    config['model_dir'] = os.path.join(config['results_dir'], config['model_dir'])
    config['img_dir'] = os.path.join(config['results_dir'], config['img_dir'])
    utils.makedirs(config['model_dir'])
    utils.makedirs(config['img_dir'])

    # set seed for all random processes
    utils.set_seed(config['seed'])

    main(config)