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


def plot_test_examples(log_samples, model, writer, config, step=0):
    model.eval()
    # apply encoding and decoding over a small subset of the training set
    imgs, labels = log_samples
    imgs = imgs.to(config['device'])
    labels = labels.to(config['device']).float()

    examples_to_show = len(imgs)

    preds, imgs_proto_recon = model.forward_single(imgs)

    # print(f"\nTest Preds: \n{np.round(preds.detach().cpu().numpy(), 2)} \n")

    recons = imgs_proto_recon.detach().cpu()
    imgs = imgs.detach().cpu()
    preds = preds.detach().cpu()

    # compare original images to their reconstructions
    n_rows = 4
    f, a = plt.subplots(n_rows, examples_to_show, figsize=(examples_to_show, n_rows))

    # set axis off for all
    [axi.set_axis_off() for axi in a.ravel()]

    a[0][0].text(0, -2, s='input', fontsize=10)
    a[1][0].text(0,-2, s='recon proto', fontsize=10)
    a[2][0].text(0,-2, s='GT categories', fontsize=10)
    a[3][0].text(0,-2, s='Pred categories', fontsize=10)

    for i in range(examples_to_show):
        a[0][i].imshow(imgs[i].permute(1, 2, 0).squeeze(),
                       cmap='gray',
                       interpolation='none')

        # convert to RGB numpy array
        recons_np = recons[i].permute(1, 2, 0).squeeze().numpy()
        # convert -1 1 range to 0 255 range for plotting
        recons_np = ((recons_np - recons_np.min())
                          * (1 / (recons_np.max() - recons_np.min()) * 255)).astype('uint8')
        a[1][i].imshow(recons_np,
                       cmap='gray',
                       interpolation='none')

        a[2][i].imshow(labels[i].unsqueeze(dim=0).detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        a[3][i].imshow(preds[i].unsqueeze(dim=0).detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)

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

    # plot symbol/prediction arrays
    if config['n_groups'] == 1:
        f, a = plt.subplots(1, 2)
    else:
        f, a = plt.subplots(1, config['n_groups'])
    # set axis off for all
    [axi.set_axis_off() for axi in a.ravel()]
    for group_id in range(config['n_groups']):
        a[group_id].text(0, -2, s=f"category {group_id+1}", fontsize=10)
        a[group_id].imshow(
            preds[:, model.attr_positions[group_id]:model.attr_positions[group_id+1]].detach().cpu().numpy(),
            cmap='gray', vmin=0, vmax=1
        )

    if writer:
        img_save_path = os.path.join(config['img_dir'], f'{step:05d}' + '_code' + '.png')
        plt.savefig(img_save_path,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        image = Image.open(img_save_path)
        image = TF.to_tensor(image)
        writer.add_image(f'train_rec/code', image, global_step=step)


    model.train()


def plot_train_examples(imgs, recon_imgs, writer, config, step=0):
    n_examples_to_show = 3

    # specific image pair ids
    for pair_idx in range(n_examples_to_show):
        f, a = plt.subplots(2, 2)
        # set axis off for all
        [axi.set_axis_off() for axi in a.ravel()]
        a = a.flat

        a[0].set_title('input', fontsize=10)
        a[1].set_title('neg', fontsize=10)
        a[2].set_title('input', fontsize=10)
        a[3].set_title('neg recon', fontsize=10)

        img0 = imgs[0] #* 2 + 0.5
        img1 = imgs[1] #* 2 + 0.5

        img0_r = recon_imgs[0][pair_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()
        img1_r = recon_imgs[1][pair_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()

        # convert -1 1 range to 0 255 range for plotting
        img0_r = ((img0_r - img0_r.min())
                          * (1 / (img0_r.max() - img0_r.min()) * 255)).astype('uint8')
        img1_r = ((img1_r - img1_r.min())
                          * (1 / (img1_r.max() - img1_r.min()) * 255)).astype('uint8')


        a[0].imshow(img0[pair_idx].reshape(config['img_shape']).permute(1, 2, 0).squeeze().detach().cpu().numpy(),
                       cmap='gray',
                       interpolation='none')
        a[1].imshow(img1[pair_idx].reshape(config['img_shape']).permute(1, 2, 0).squeeze().detach().cpu().numpy(),
                       cmap='gray',
                       interpolation='none')

        a[2].imshow(img0_r, cmap='gray', interpolation='none')
        a[3].imshow(img1_r, cmap='gray', interpolation='none')

        # if writer:
        img_save_path = os.path.join(config['img_dir'], f'{step:05d}' + f'_train_decoding_result_pair_{pair_idx}.png')
        plt.savefig(img_save_path,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        image = Image.open(img_save_path)
        image = TF.to_tensor(image)
        writer.add_image(f"train_rec/train_decoding_result_pair_{pair_idx}", image, global_step=step)


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


def convert_one_hot_to_ids(one_hot, group_ids):
    ids = torch.cat(
        [torch.argmax(one_hot[:, group_ids[i]:group_ids[i + 1]], dim=1, keepdim=True)
         for i in range(len(group_ids)-1)],
        dim=1
    )
    return ids
