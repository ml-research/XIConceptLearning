import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
import itertools
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import accuracy_score


class WrappedModel(nn.Module):
	def __init__(self, module):
		super(WrappedModel, self).__init__()
		self.module = module # that I actually define.
	def forward(self, x):
		return self.module(x)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        tmp = tensor.clone()
        for t, m, s in zip(tmp, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return torch.clamp(tmp, 0, 1)


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
    model.eval()

    # contains start and end id of the groups
    n_proto_vecs = config['prototype_vectors']
    n_proto_vecs = list(np.cumsum(n_proto_vecs))
    n_proto_vecs.insert(0, 0)

    # create all possible id combinations between prototype groups
    comb_proto_ids = np.array(list(itertools.product(*[np.arange(i) for i in config['prototype_vectors']])))
    n_comb = len(comb_proto_ids)

    # turn into one hot
    comb_one_hot = torch.zeros(len(comb_proto_ids), n_proto_vecs[-1], device=config['device'])
    for group_id in range(model.n_groups):
        comb_one_hot[range(n_comb), n_proto_vecs[group_id] + comb_proto_ids[:, group_id]] = 1.

    # forward one hot representation
    proto_imgs = model.proto_decode(comb_one_hot).detach().cpu()

    fig, ax = plt.subplots(nrows=int(np.ceil(np.sqrt(n_comb))), ncols=int(np.ceil(np.sqrt(n_comb))))
    ax = ax.flatten()
    # set axis off for all
    [axi.set_axis_off() for axi in ax.ravel()]

    # visualize the prototype images
    for idx in range(n_comb):
        # convert to RGB numpy array
        proto_recon_np = proto_imgs[idx].reshape(config['img_shape']).permute(1, 2, 0).squeeze().numpy()
        # convert -1 1 range to 0 255 range for plotting
        proto_recon_np = ((proto_recon_np - proto_recon_np.min())
                          * (1 / (proto_recon_np.max() - proto_recon_np.min()) * 255)).astype('uint8')
        ax[idx].imshow(proto_recon_np, interpolation='none')

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

    model.train()


def plot_examples_code(imgs, recons, preds, writer, config, step=0):
    n_examples_to_show = 8

    fig, ax = plt.subplots(n_examples_to_show, 3, figsize=(10, 15))
    # set axis off for all
    [axi.set_axis_off() for axi in ax.ravel()]

    for idx in range(n_examples_to_show):

        ax[idx, 0].imshow(np.clip(np.moveaxis(imgs[idx].detach().cpu().numpy(), [0, 1, 2], [2, 0, 1]),
                             0, 1))
        ax[idx, 1].imshow(np.clip(np.moveaxis(recons[idx].detach().cpu().numpy(),
                                         [0, 1, 2], [2, 0, 1]),
                             0, 1
                             ))
        ax[idx, 2].imshow(preds[idx].detach().cpu().numpy(), cmap='gray')
    # if writer:
    img_save_path = os.path.join(config['img_dir'], f'{step:05d}' + f'_train_decoding_test_result.png')
    fig.savefig(img_save_path)
    plt.close()

    image = Image.open(img_save_path)
    image = TF.to_tensor(image)
    writer.add_image(f"train_rec/train_decoding_result_{idx}", image, global_step=step)


def plot_examples(imgs, recons, writer, config, step=0):

    n_examples_to_show = 4

    fig, ax = plt.subplots(n_examples_to_show, 3, figsize=(10, 15))
    # set axis off for all
    [axi.set_axis_off() for axi in ax.ravel()]

    for idx in range(n_examples_to_show):
        ax[idx, 0].imshow(np.clip(np.moveaxis(imgs[idx].detach().cpu().numpy(), [0, 1, 2], [2, 0, 1]),
                             0, 1))
        ax[idx, 1].imshow(np.clip(np.moveaxis(recons[idx].detach().cpu().numpy(),
                                         [0, 1, 2], [2, 0, 1]),
                             0, 1
                             ))
    # if writer:
    img_save_path = os.path.join(config['img_dir'], f'{step:05d}' + f'_train_decoding_test_result.png')
    fig.savefig(img_save_path)
    plt.close()

    image = Image.open(img_save_path)
    image = TF.to_tensor(image)
    writer.add_image(f"train_rec/train_decoding_test_result", image, global_step=step)


def makedirs(path):
    """
    If path does not exist in the file system, create it
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def multioutput_multiclass_acc(gt, pred):
    res = []
    for i in range(len(gt)):
        res.append(accuracy_score(gt[i], pred[i]))
    return np.mean(res)


def convert_one_hot_to_ids(one_hot, group_ids):
    ids = torch.cat(
        [torch.argmax(one_hot[:, group_ids[i]:group_ids[i + 1]], dim=1, keepdim=True)
         for i in range(len(group_ids)-1)],
        dim=1
    )
    return ids


def gen_one_hot_from_ids(n_samples, length, idx, device):
    one_hot = torch.zeros((n_samples, length), device=device)
    one_hot[range(n_samples), idx] = 1.
    return one_hot