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
    # create all possible id combinations between prototype groups
    comb_proto_ids = list(itertools.product(*[np.arange(i) for i in config['prototype_vectors']]))
    # turn into dict of lists for
    comb_proto_ids = [{k: [l[k]] for k in range(config['n_prototype_groups'])} for l in comb_proto_ids]

    fig, ax = plt.subplots(nrows=int(np.ceil(np.sqrt(len(comb_proto_ids)))),
                           ncols=int(np.ceil(np.sqrt(len(comb_proto_ids)))))
    ax = ax.flatten()

    for idx, comb_proto_id in enumerate(comb_proto_ids):
        agg_proto = model.comp_combined_prototype_per_sample(comb_proto_id)
        prototype_img = model.dec_wrapper(agg_proto).detach().cpu()

        # visualize the prototype images
        ax[idx].imshow(prototype_img.reshape(config['img_shape']).permute(1, 2, 0).squeeze(),
                   # config['img_shape'][1], config['img_shape'][2]
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
    # apply encoding and decoding over a small subset of the training set
    if type(log_samples) is tuple:
        imgs, labels = log_samples
    else:
        imgs = log_samples
    examples_to_show = len(imgs)

    if rec_protos is None:
        if config['learn'] == 'weakly':
            res_dict = model.forward_single(imgs)
            rec_z = res_dict['recon_imgs']
            rec_protos = res_dict['recon_protos']
        elif config['learn'] == 'unsup':
            res_dict = model.forward(imgs)
            rec_z = res_dict['recon_imgs']
            rec_protos = res_dict['recon_protos']

    rec_z = rec_z.detach().cpu()
    rec_protos = rec_protos.detach().cpu()

    imgs = imgs.detach().cpu()

    # compare original images to their reconstructions
    n_rows = 3
    f, a = plt.subplots(n_rows, examples_to_show, figsize=(examples_to_show, n_rows))

    a[0][0].text(0, -2, s='input', fontsize=10)
    a[1][0].text(0,-2, s='recon z', fontsize=10)
    # a[2][0].text(0,-2, s='recon min proto', fontsize=10)
    a[2][0].text(0, -2, s='agg proto', fontsize=10)

    for i in range(examples_to_show):
        a[0][i].imshow(imgs[i].reshape(config['img_shape']).permute(1, 2, 0).squeeze(),
                       cmap='gray',
                       interpolation='none')
        a[0][i].axis('off')

        a[1][i].imshow(rec_z[i].reshape(config['img_shape']).permute(1, 2, 0).squeeze(),
                       cmap='gray',
                       interpolation='none')
        a[1][i].axis('off')

        a[2][i].imshow(rec_protos[i].reshape(config['img_shape']).permute(1, 2, 0).squeeze(),
                       cmap='gray',
                       interpolation='none')
        a[2][i].axis('off')

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
