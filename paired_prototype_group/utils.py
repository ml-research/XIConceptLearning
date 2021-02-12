import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF
from PIL import Image


def makedirs(path):
    """
    If path does not exist in the file system, create it
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


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
    feature_vectors_z = res_dict['latent_enc']
    prototype_vectors = res_dict['proto_vecs']
    mixed_prototypes = res_dict['agg_protos']
    pair_s_weights = res_dict['pair_s_weights']
    return rec_imgs, rec_protos, dists, feature_vectors_z, prototype_vectors, mixed_prototypes, pair_s_weights


def plot_prototypes(model, prototype_vectors, writer, e, config):
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

        img_save_path = os.path.join(config['img_dir'], f'{e:05d}' + f'_group_{group_id}'+ '_prototype_result' + '.png')
        plt.savefig(img_save_path,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        image = Image.open(img_save_path)
        image = TF.to_tensor(image)
        writer.add_image(f'train_proto/group{group_id}', image, global_step=e)


def plot_examples(log_samples, model, writer, e, config):
    # apply encoding and decoding over a small subset of the training set
    imgs = log_samples
    examples_to_show = len(log_samples)

    _, rec_protos, _, _, _, _ = model.forward(imgs[:examples_to_show], std=0)
    rec_protos = rec_protos.detach().cpu()

    imgs = imgs.detach().cpu()

    # compare original images to their reconstructions
    n_rows = 2
    f, a = plt.subplots(n_rows, examples_to_show, figsize=(examples_to_show, n_rows))

    a[0][0].text(0, -2, s='input', fontsize=10)
    # a[1][0].text(0,-2, s='recon z', fontsize=10)
    # a[2][0].text(0,-2, s='recon min proto', fontsize=10)
    a[1][0].text(0, -2, s='recon softmin proto', fontsize=10)

    for i in range(examples_to_show):
        a[0][i].imshow(imgs[i].reshape(config['img_shape']).permute(1, 2, 0).squeeze(),
                       cmap='gray',
                       interpolation='none')
        a[0][i].axis('off')

        a[1][i].imshow(rec_protos[i].reshape(config['img_shape']).permute(1, 2, 0).squeeze(),
                       cmap='gray',
                       interpolation='none')
        a[1][i].axis('off')

    img_save_path = os.path.join(config['img_dir'], f'{e:05d}' + '_decoding_result' + '.png')
    plt.savefig(img_save_path,
                transparent=True,
                bbox_inches='tight',
                pad_inches=0)
    plt.close()

    image = Image.open(img_save_path)
    image = TF.to_tensor(image)
    writer.add_image(f'train_rec/decoding_result', image, global_step=e)
