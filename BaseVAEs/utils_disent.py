import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import accuracy_score

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


def get_cum_group_ids(self):
    group_ids = list(np.cumsum(self.n_proto_vecs))
    group_ids.insert(0, 0)
    group_ranges = []
    for k in range(self.n_proto_groups):
        group_ranges.append([group_ids[k], group_ids[k+1]])
    return group_ranges


def freeze_enc(model):
    for name, p in model.named_parameters():
        if "enc" in name:
            p.requires_grad = False


def unfold_res_dict(res_dict):
    """
    Takes results dict from model forward pass, unwraps all variables and returns these.
    :param res_dict: dict from model forward pass
    :return: all result variable
    """
    rec_imgs = res_dict['recon_imgs']
    rec_protos = res_dict['recon_protos']
    attr_probs = res_dict['attr_prob_pairs']
    feature_vecs_z = res_dict['latent_enc']
    proto_vecs = res_dict['proto_vecs']
    agg_protos = res_dict['agg_protos']
    return rec_imgs, rec_protos, attr_probs, feature_vecs_z, proto_vecs, agg_protos


def plot_prototypes(model, writer, config, step=0):
    """
    Visualize all mixtures of prototypes.
    :param model:
    :param prototype_vectors:
    :param writer:
    :param config:
    :param step:
    :return:
    """
    # create list of number of prototypes per group
    n_proto_vecs = [config['n_protos'] for i in range(config['n_groups'])]
    # create all possible id combinations between prototype groups
    comb_proto_ids = list(itertools.product(*[np.arange(i) for i in n_proto_vecs]))
    # turn into dict of lists for
    z_selection = torch.zeros((len(comb_proto_ids), config['n_groups'], config['n_protos']), device=config['device'])
    for i, ids in enumerate(comb_proto_ids):
        for j in range(len(ids)):
            z_selection[i, j, ids[j]] = 1.

    fig, ax = plt.subplots(nrows=int(np.ceil(np.sqrt(len(comb_proto_ids)))),
                           ncols=int(np.ceil(np.sqrt(len(comb_proto_ids)))))
    ax = ax.flatten()

    prototype_img = model.dec_proto_by_selection(z_selection).detach().cpu()

    # visualize the prototype images
    for idx in range(len(comb_proto_ids)):
        # convert to RGB numpy array
        proto_recon_np = prototype_img[idx].reshape(config['img_shape']).permute(1, 2, 0).squeeze().numpy()
        # convert -1 1 range to 0 255 range for plotting
        proto_recon_np = ((proto_recon_np - proto_recon_np.min())
                          * (1 / (proto_recon_np.max() - proto_recon_np.min()) * 255)).astype('uint8')
        ax[idx].imshow(proto_recon_np,
                   interpolation='none')
        ax[idx].axis('off')

    if writer:
        img_save_path = os.path.join(config['img_dir'],
                                     f'{step:05d}' + '_comb_prototypeslot_result' + '.png')

        plt.savefig(img_save_path,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        image = Image.open(img_save_path)
        image = TF.to_tensor(image)
        writer.add_image(f'train_proto', image, global_step=step)


def plot_examples(log_samples, model, writer, config, step=0, rec_protos=None):
    model.eval()
    # apply encoding and decoding over a small subset of the training set
    if type(log_samples) is tuple:
        imgs, labels = log_samples
    else:
        imgs = log_samples
    examples_to_show = len(imgs)

    if rec_protos is None:
        recons, z = model.forward(imgs)

    recons = recons.detach().cpu()
    imgs = imgs.detach().cpu()
    z = z.detach().cpu()

    # compare original images to their reconstructions
    n_rows = 3
    f, a = plt.subplots(n_rows, examples_to_show, figsize=(examples_to_show, n_rows))

    a[0][0].text(0, -2, s='input', fontsize=10)
    a[1][0].text(0,-2, s='recon z', fontsize=10)
    a[2][0].text(0,-2, s='z categories', fontsize=10)
    # a[2][0].text(0, -2, s='recon agg proto', fontsize=10)

    for i in range(examples_to_show):
        a[0][i].imshow(imgs[i].reshape(config['img_shape']).permute(1, 2, 0).squeeze(),
                       cmap='gray',
                       interpolation='none')
        a[0][i].axis('off')

        # convert to RGB numpy array
        recons_np = recons[i].reshape(config['img_shape']).permute(1, 2, 0).squeeze().numpy()
        # convert -1 1 range to 0 255 range for plotting
        recons_np = ((recons_np - recons_np.min())
                          * (1 / (recons_np.max() - recons_np.min()) * 255)).astype('uint8')
        a[1][i].imshow(recons_np,
                       cmap='gray',
                       interpolation='none')
        a[1][i].axis('off')

        img = a[2][i].imshow(z[i].numpy(),
                       cmap='gray',
                       vmin=0., vmax=1.)
        a[2][i].axis('off')
        if i == examples_to_show-1:
            f.colorbar(img)

        # # convert to RGB numpy array
        # recons_proto_np = recons_proto[i].reshape(config['img_shape']).permute(1, 2, 0).squeeze().numpy()
        # # convert -1 1 range to 0 255 range for plotting
        # recons_proto_np = ((recons_proto_np - recons_proto_np.min())
        #                   * (1 / (recons_proto_np.max() - recons_proto_np.min()) * 255)).astype('uint8')
        # a[2][i].imshow(recons_proto_np,
        #                cmap='gray',
        #                interpolation='none')
        # a[2][i].axis('off')

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
    model.train()


def plot_train_examples(batch, model, writer, config, step=0, rec_protos=None):
    model.eval()

    imgs0, imgs1 = batch[0]
    imgs0 = imgs0.to(config['device'])
    imgs1 = imgs1.to(config['device'])

    recons0, z0 = model.forward(imgs0)
    recons1, z1 = model.forward(imgs1)

    recons0 = recons0.detach().cpu()
    recons1 = recons1.detach().cpu()
    imgs0 = imgs0.detach().cpu()
    imgs1 = imgs1.detach().cpu()
    z0 = z0.detach().cpu()
    z1 = z1.detach().cpu()

    recons = (recons0, recons1)
    imgs = (imgs0, imgs1)
    z = (z0, z1)

    n_examples_to_show = 2
    n_rows = 3
    n_cols = n_examples_to_show * 2
    f, a = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))

    # specific image pair ids
    for i, pair_idx in enumerate([0, 5]):

        a[0, i*2].set_title('input', fontsize=10)
        a[0, i*2].set_title('recon z', fontsize=10)
        a[0, i*2].set_title('z categories', fontsize=10)

        for img_idx in range(2):
            col_idx = img_idx + i*2
            a[0, col_idx].imshow(imgs[img_idx][pair_idx].reshape(config['img_shape']).permute(1, 2, 0).squeeze(),
                           cmap='gray',
                           interpolation='none')
            a[0, col_idx].axis('off')

            # convert to RGB numpy array
            recons_np = recons[img_idx][pair_idx].reshape(config['img_shape']).permute(1, 2, 0).squeeze().numpy()
            # convert -1 1 range to 0 255 range for plotting
            recons_np = ((recons_np - recons_np.min())
                              * (1 / (recons_np.max() - recons_np.min()) * 255)).astype('uint8')
            a[1, col_idx].imshow(recons_np,
                           cmap='gray',
                           interpolation='none')
            a[1, col_idx].axis('off')

            img = a[2, col_idx].imshow(z[img_idx][pair_idx].numpy(),
                           cmap='gray',
                           vmin=0., vmax=1.)
            a[2, col_idx].axis('off')
            if col_idx == n_cols-1:
                f.colorbar(img)

    if writer:
        img_save_path = os.path.join(config['img_dir'], f'{step:05d}' + f'_train_decoding_result_pair' + '.png')
        plt.savefig(img_save_path,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        image = Image.open(img_save_path)
        image = TF.to_tensor(image)
        writer.add_image(f"train_rec/train_decoding_result_pair", image, global_step=step)
    model.train()


def makedirs(path):
    """
    If path does not exist in the file system, create it
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def comp_multilabel_acc(attr_probs, labels, group_ranges):
    attr_preds = np.concatenate((one_hot_to_ids_list(attr_probs[0], group_ranges),
                                 one_hot_to_ids_list(attr_probs[1], group_ranges)), axis=0)
    attr_gt = one_hot_to_ids_list(labels, group_ranges)
    return multioutput_multiclass_acc(attr_gt, attr_preds)


def multioutput_multiclass_acc(gt, pred):
    res = []
    for i in range(len(gt)):
        res.append(accuracy_score(gt[i], pred[i]))
    return np.mean(res)


def one_hot_to_ids_list(attr_prob, group_ranges):
    attr_ids = np.ones((len(group_ranges), attr_prob.shape[0]))
    for k, ids in enumerate(group_ranges):
        attr_ids[k, :] = torch.argmax(attr_prob[:, ids[0]:ids[1]], dim=1).detach().cpu()
    attr_ids = attr_ids.T
    return attr_ids
