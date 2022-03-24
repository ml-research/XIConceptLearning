import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib
from torchvision import transforms
import pickle

matplotlib.use('Agg')
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from rtpt.rtpt import RTPT
from torch.nn.parameter import Parameter

import experiments.ProtoLearning.utils as utils
import experiments.ProtoLearning.data as data
from experiments.ProtoLearning.models.icsn import iCSN
from experiments.ProtoLearning.args import parse_args_as_dict

def get_new_attr_id(seed):
    if seed == 0:
        # category id for which to add a new attribute
        CAT_ID = 1
        # attribute ids which are already binded to within that category
        BINDED_ATTR_IDS = [0, 1, 3, 5]
        # attribute id for new attribute
        NEW_SHAPE_ATTR_ID = 2
    elif seed == 1:
        # category id for which to add a new attribute
        CAT_ID = 1
        # attribute ids which are already binded to within that category
        BINDED_ATTR_IDS = [1, 2, 3, 4]
        # attribute id for new attribute
        NEW_SHAPE_ATTR_ID = 5
    elif seed == 3:
        # category id for which to add a new attribute
        CAT_ID = 1
        # attribute ids which are already binded to within that category
        BINDED_ATTR_IDS = [1, 2, 4, 5]
        # attribute id for new attribute
        NEW_SHAPE_ATTR_ID = 3
    elif seed == 13:
        # category id for which to add a new attribute
        CAT_ID = 1
        # attribute ids which are already binded to within that category
        BINDED_ATTR_IDS = [0, 1, 4, 5]
        # attribute id for new attribute
        NEW_SHAPE_ATTR_ID = 2
    elif seed == 21:
        # category id for which to add a new attribute
        CAT_ID = 1
        # attribute ids which are already binded to within that category
        BINDED_ATTR_IDS = [1, 2, 4, 5]
        # attribute id for new attribute
        NEW_SHAPE_ATTR_ID = 0
    return CAT_ID, BINDED_ATTR_IDS, NEW_SHAPE_ATTR_ID


def get_novel_concept_data(model, config, CAT_ID, BINDED_ATTR_IDS, NEW_SHAPE_ATTR_IDS):
    # get new shape data
    new_shape_np = np.load(os.path.join(f"{config['data_dir']}", "train_ecr_newshape.npy"))
    new_shape_np = (new_shape_np - new_shape_np.min()) / (new_shape_np.max() - new_shape_np.min())
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    new_shape_imgs = torch.empty(new_shape_np.shape, device=config['device'])
    new_shape_imgs = new_shape_imgs.permute(0, 3, 1, 2)
    for i in range(new_shape_imgs.shape[0]):
        new_shape_imgs[i] = transform(np.uint8(new_shape_np[i] * 255)).float()

    labels_fp = os.path.join(f"{config['data_dir']}","train_ecr_newshape_labels.pkl")
    with open(labels_fp, 'rb') as f:
        labels_dict = pickle.load(f)
        new_shape_labels = labels_dict['labels_one_hot']
        new_shape_labels_as_id = labels_dict['labels']

    # mix the new shape data with some additional old trainig data to prevent forgetting
    # get train data
    config['batch_size'] = 512
    config['img_shape'] = (3, 64, 64)
    train_dataset = data.ECR_PairswithTest(config['data_dir'],
                                                   attrs='')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # get a single batch from the previous training data
    batch = next(iter(train_loader))

    train_imgs, _, _, _ = batch
    train_imgs = train_imgs[0].to(config['device'])

    # let model predict the labels for all previous training samples
    preds, recons = model.forward_single(train_imgs)
    gt_train_labels_one_hot = torch.clone(preds)
    gt_train_labels_one_hot[gt_train_labels_one_hot > 0.5] = 1.
    gt_train_labels_one_hot[gt_train_labels_one_hot <= 0.5] = 0.

    # let model predict the labels for the new train imgs and replace the labels for the relevant
    # new shape category with a desired label
    preds, recons = model.forward_single(new_shape_imgs)

    # GT one hot encoding stating which attribute the new shape should be predicted as
    gt_one_hot_new_shape = torch.zeros(config['prototype_vectors'][CAT_ID])
    gt_one_hot_new_shape[NEW_SHAPE_ATTR_IDS] = 1.
    gt_new_shape_labels_one_hot = torch.clone(preds)
    gt_new_shape_labels_one_hot[gt_new_shape_labels_one_hot > 0.5] = 1.
    gt_new_shape_labels_one_hot[gt_new_shape_labels_one_hot <= 0.5] = 0.
    gt_new_shape_labels_one_hot[:, model.attr_positions[CAT_ID]:model.attr_positions[CAT_ID+1]] = gt_one_hot_new_shape

    new_train_imgs = torch.cat([train_imgs, new_shape_imgs], dim=0)
    new_train_labels_as_id = torch.cat([gt_train_labels_one_hot.detach(), gt_new_shape_labels_one_hot.detach()], dim=0)

    new_train_dataset = torch.utils.data.TensorDataset(new_train_imgs, new_train_labels_as_id)
    new_train_dataloader = torch.utils.data.DataLoader(new_train_dataset, batch_size=128,
                                                      shuffle=True)
    return new_train_dataloader


def train(model, data_loader, log_samples, optimizer, scheduler, writer, cur_epoch, config):

    rtpt = RTPT(name_initials=config['initials'], experiment_name='XIC_PrototypeDL', max_iterations=config['epochs'])
    rtpt.start()

    model.train()
    model.softmax_temp = 1.

    for e in range(0, 600):
        start = time.time()

        loss_dict = dict(
            {'loss': 0, "recon_loss": 0, "binding_loss": 0})

        torch.autograd.set_detect_anomaly(True)

        for i, batch in enumerate(data_loader):

            imgs, gt_labels_one_hot = batch
            imgs = imgs.to(config['device'])
            gt_labels_one_hot = gt_labels_one_hot.to(config['device'])
            batch_size = imgs.shape[0]

            # step scheduler for soft max temp
            if e >= 200:
                model.softmax_temp = .0001
            elif e >= 150:
                model.softmax_temp = .001
            elif e >= 100:
                model.softmax_temp = .01
            elif e >= 50:
                model.softmax_temp = .1

            # turn reconstruction loss on after certain epochs
            if e >= 300:
                config['lambda_recon_proto'] = 1.

            # [B, N_Protos], [B, 3, W, H]
            preds, recons = model.forward_single(imgs)

            ave_recon_loss = F.mse_loss(recons, imgs)

            ave_binding_loss = F.smooth_l1_loss(preds, gt_labels_one_hot, reduction="mean")


            loss = config['lambda_recon_proto'] * ave_recon_loss + \
                    config['lambda_binding_loss'] * ave_binding_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_dict['recon_loss'] += ave_recon_loss.item() if config['lambda_recon_proto'] > 0. else 0.
            loss_dict['binding_loss'] += ave_binding_loss.item() if config['lambda_binding_loss'] > 0. else 0.
            loss_dict['loss'] += loss.item()

        for key in loss_dict.keys():
            loss_dict[key] /= len(data_loader)

        if (e + 1) % config['print_step'] == 0 or e == config['epochs'] - 1:
            print(f'epoch {e} - loss {loss.item():2.4f} - time/epoch {(time.time() - start):2.2f}'
              f'- softmax_temp {model.softmax_temp}')
            loss_summary = ''
            for key in loss_dict.keys():
                loss_summary += f'{key} {loss_dict[key]:2.4f} '
            print(loss_summary)

    state = {
        'model': model.cpu().state_dict(),
        'model_misc': {'prototypes': model.cpu().proto_dict,
                       'softmax_temp': model.cpu().softmax_temp},
        'optimizer': optimizer.state_dict(),
        'ep': e,
        'config': config
    }
    # torch.save(state, os.path.join(writer.log_dir, f"{config['exp_name']}.pth"))
    torch.save(state, os.path.join(config['model_dir'], '%05d.pth' % (e)))


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
        ckpt = torch.load(config['ckpt_fp'], map_location=torch.device(config['device']))
        cur_epoch = ckpt['ep']
        _model.load_state_dict(ckpt['model'])
        _model.proto_dict = ckpt['model_misc']['prototypes']
        _model.softmax_temp = ckpt['model_misc']['softmax_temp']
        print(f"loading {config['ckpt_fp']} from epoch {cur_epoch} for further training")

    # get the new attribute id, based on the seed
    CAT_ID, BINDED_ATTR_IDS, NEW_SHAPE_ATTR_IDS = get_new_attr_id(config['seed'])

    # get novel concept data set, with gt labels predicted by the model itself, except for novel concept
    new_train_dataloader = get_novel_concept_data(_model, config, CAT_ID, BINDED_ATTR_IDS, NEW_SHAPE_ATTR_IDS)

    config['print_step'] = 20
    config['lambda_recon_proto'] = 0.
    config['lambda_binding_loss'] = 10.
    config['learning_rate'] = 0.00005

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
        train(_model, new_train_dataloader, test_set, optimizer, scheduler, writer, cur_epoch, config)
    else:
        test(_model, test_set, writer, config)

if __name__ == '__main__':
    # get config
    config = parse_args_as_dict(sys.argv[1:])

    main(config)
