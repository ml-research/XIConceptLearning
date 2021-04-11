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


def makedirs(path):
    """
    If path does not exist in the file system, create it
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


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


def plot_train_examples(imgs, distances, writer, config, step=0):
    n_examples_to_show = 1

    distances_aug = distances[0].permute(1, 0)
    distances_neg = distances[1].permute(1, 0)

    # specific image pair ids
    for i, triplet_idx in enumerate([0, n_examples_to_show]):
        f, a = plt.subplots(2, 3)

        a = a.flat

        a[0].set_title('input', fontsize=10)
        a[1].set_title('augment', fontsize=10)
        a[2].set_title('negative ', fontsize=10)
        a[4].set_title('distances augm', fontsize=10)
        a[5].set_title('distances negative', fontsize=10)

        img = imgs[0] * 2 + 0.5
        a_img = imgs[1] * 2 + 0.5
        neg_img = imgs[2] * 2 + 0.5

        a[0].imshow(img[triplet_idx].reshape(config['img_shape']).permute(1, 2, 0).squeeze().detach().cpu().numpy(),
                       cmap='gray',
                       interpolation='none')
        a[0].axis('off')
        a[1].imshow(a_img[triplet_idx].reshape(config['img_shape']).permute(1, 2, 0).squeeze().detach().cpu().numpy(),
                       cmap='gray',
                       interpolation='none')
        a[1].axis('off')
        a[2].imshow(neg_img[triplet_idx].reshape(config['img_shape']).permute(1, 2, 0).squeeze().detach().cpu().numpy(),
                       cmap='gray',
                       interpolation='none')
        a[2].axis('off')

        img = a[4].imshow(distances_aug[triplet_idx].unsqueeze(dim=1).detach().cpu().numpy(),
                       cmap='gray',
                       vmin=-1., vmax=1.)
        a[4].axis('off')

        img = a[5].imshow(distances_neg[triplet_idx].unsqueeze(dim=1).detach().cpu().numpy(),
                       cmap='gray',
                       vmin=-1., vmax=1.)
        a[5].axis('off')

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
