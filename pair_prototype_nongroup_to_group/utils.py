import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF
from PIL import Image


def set_seed(seed=42):
    """
    Set random seeds for all possible random processes.
    :param seed: int
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unfold_res_dict(res_dict):
    """
    Takes results dict from model forward pass, unwraps all variables and returns these.
    :param res_dict: dict from model forward pass
    :return: all result variable
    """
    rec_imgs = res_dict['recon_imgs']
    rec_protos = res_dict['recon_protos']
    dists = res_dict['dists']
    s_weights = res_dict['s_weights']
    feature_vecs_z = res_dict['latent_enc']
    proto_vecs = res_dict['proto_vecs']
    agg_protos = res_dict['agg_protos']
    pair_s_weights = res_dict['pair_s_weights']
    # the individual s or min_weights of the img pairs, rather then concatenated as in s_weights
    if 's_weights_pairs' in res_dict.keys():
        s_weights_pair = res_dict['s_weights_pairs']
        return rec_imgs, rec_protos, dists, s_weights, feature_vecs_z, proto_vecs, agg_protos, pair_s_weights, s_weights_pair
    elif 'dists_pairs' in res_dict.keys():
        dists_pair = res_dict['dists_pairs']
        return rec_imgs, rec_protos, dists, s_weights, feature_vecs_z, proto_vecs, agg_protos, pair_s_weights, dists_pair
    else:
        return rec_imgs, rec_protos, dists, s_weights, feature_vecs_z, proto_vecs, agg_protos, pair_s_weights


def plot_prototypes(model, prototype_vectors, writer, config, step=0):
    # decode uncombined prototype vectors
    for group_id in range(config['n_prototype_groups']):
        prototype_imgs = model.dec_prototypes(
            prototype_vectors[group_id]).detach().cpu()

        # visualize the prototype images
        cnt = 321
        for p in prototype_imgs:
            plt.subplot(cnt)
            plt.imshow(p.reshape(config['img_shape']).permute(1, 2, 0).squeeze(),
                       # config['img_shape'][1], config['img_shape'][2]
                       cmap='gray',
                       interpolation='none')
            plt.axis('off')
            cnt += 1
        if writer:
            img_save_path = os.path.join(config['img_dir'],
                                         f'{step:05d}' + f'_group_{group_id}' + '_prototype_result' + '.png')

            plt.savefig(img_save_path,
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()

            image = Image.open(img_save_path)
            image = TF.to_tensor(image)
            writer.add_image(f'train_proto/group{group_id}', image, global_step=step)


def plot_examples(log_samples, model, writer, config, step=0, rec_protos=None):
    # apply encoding and decoding over a small subset of the training set
    imgs = log_samples
    examples_to_show = len(log_samples)

    if rec_protos is None:
        if config['learn'] == 'weakly':
            res_dict = model.forward_single(imgs[:examples_to_show], std=0)
            rec_protos = res_dict['recon_protos']
        elif config['learn'] == 'unsup':
            res_dict = model.forward(imgs[:examples_to_show], std=0)
            rec_protos = res_dict['recon_protos']

    rec_protos = rec_protos.detach().cpu()

    imgs = imgs.detach().cpu()

    # compare original images to their reconstructions
    n_rows = 2
    f, a = plt.subplots(n_rows, examples_to_show, figsize=(examples_to_show, n_rows))

    a[0][0].text(0, -2, s='input', fontsize=10)
    # a[1][0].text(0,-2, s='recon z', fontsize=10)
    # a[2][0].text(0,-2, s='recon min proto', fontsize=10)
    a[1][0].text(0, -2, s='agg proto', fontsize=10)

    for i in range(examples_to_show):
        a[0][i].imshow(imgs[i].reshape(config['img_shape']).permute(1, 2, 0).squeeze(),
                       cmap='gray',
                       interpolation='none')
        a[0][i].axis('off')

        a[1][i].imshow(rec_protos[i].reshape(config['img_shape']).permute(1, 2, 0).squeeze(),
                       cmap='gray',
                       interpolation='none')
        a[1][i].axis('off')

    if writer:
        img_save_path = os.path.join(config['img_dir'], f'{step:05d}' + '_decoding_result' + '.png')
        plt.savefig(img_save_path,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        image = Image.open(img_save_path)
        image = TF.to_tensor(image)
        writer.add_image(f'train_rec/decoding_result', image, global_step=step)


def makedirs(path):
    """
    If path does not exist in the file system, create it
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
