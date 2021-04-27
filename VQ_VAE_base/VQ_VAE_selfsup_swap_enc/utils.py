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


def plot_train_examples(imgs, recon_imgs, writer, config, step=0):
    n_examples_to_show = 3

    # specific image pair ids
    for triplet_idx in range(n_examples_to_show):
        f, a = plt.subplots(2, 3)

        a = a.flat

        a[0].set_title('input', fontsize=10)
        a[1].set_title('augm', fontsize=10)
        a[2].set_title('neg', fontsize=10)
        a[3].set_title('input', fontsize=10)
        a[4].set_title('augm recon', fontsize=10)
        a[5].set_title('neg recon', fontsize=10)

        img = imgs[0] * 2 + 0.5
        a_img = imgs[1] * 2 + 0.5
        neg_img = imgs[2] * 2 + 0.5

        img_r = recon_imgs[0][triplet_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()
        a_img_r = recon_imgs[1][triplet_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()
        neg_img_r = recon_imgs[2][triplet_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()

        # convert -1 1 range to 0 255 range for plotting
        img_r = ((img_r - img_r.min())
                          * (1 / (img_r.max() - img_r.min()) * 255)).astype('uint8')
        a_img_r = ((a_img_r - a_img_r.min())
                          * (1 / (a_img_r.max() - a_img_r.min()) * 255)).astype('uint8')
        neg_img_r = ((neg_img_r - neg_img_r.min())
                          * (1 / (neg_img_r.max() - neg_img_r.min()) * 255)).astype('uint8')


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

        a[3].imshow(img_r,
                       cmap='gray',
                       interpolation='none')
        a[3].axis('off')
        a[4].imshow(a_img_r,
                       cmap='gray',
                       interpolation='none')
        a[4].axis('off')
        a[5].imshow(neg_img_r,
                       cmap='gray',
                       interpolation='none')
        a[5].axis('off')

        # if writer:
        img_save_path = os.path.join(config['img_dir'], f'{step:05d}' + f'_train_decoding_result_pair_{triplet_idx}.png')
        plt.savefig(img_save_path,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        image = Image.open(img_save_path)
        image = TF.to_tensor(image)
        writer.add_image(f"train_rec/train_decoding_result_pair_{triplet_idx}", image, global_step=step)


def plot_train_examples_2(imgs, recon_imgs, distances, emb_distances, writer, config, batch, step=0):
    n_examples_to_show = 2

    distances_pos_neg = distances[0].permute(1, 0)
    distances_pos_aug = distances[1].permute(1, 0)
    distances_aug_neg = distances[2].permute(1, 0)

    emb_distances_pos = emb_distances[0]
    emb_distances_aug = emb_distances[1]
    emb_distances_neg = emb_distances[2]

    # specific image pair ids
    for triplet_idx in range(n_examples_to_show):
        f, a = plt.subplots(4, 3)

        a = a.flat

        a[0].set_title('input', fontsize=10)
        a[1].set_title('augm', fontsize=10)
        a[2].set_title('neg', fontsize=10)
        a[3].set_title('input', fontsize=10)
        a[4].set_title('augm recon', fontsize=10)
        a[5].set_title('neg recon', fontsize=10)
        a[6].set_title('dist pos neg', fontsize=10)
        a[7].set_title('dist pos aug', fontsize=10)
        a[8].set_title('dist aug neg', fontsize=10)
        a[9].set_title('dist pos emb', fontsize=10)
        a[10].set_title('dist aug emb', fontsize=10)
        a[11].set_title('dist neg emb', fontsize=10)

        # unnormalize
        img = imgs[0] * 2 + 0.5
        a_img = imgs[1] * 2 + 0.5
        neg_img = imgs[2] * 2 + 0.5

        img_r = recon_imgs[0][triplet_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()
        a_img_r = recon_imgs[1][triplet_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()
        neg_img_r = recon_imgs[2][triplet_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()

        # convert -1 1 range to 0 255 range for plotting
        img_r = ((img_r - img_r.min())
                          * (1 / (img_r.max() - img_r.min()) * 255)).astype('uint8')
        a_img_r = ((a_img_r - a_img_r.min())
                          * (1 / (a_img_r.max() - a_img_r.min()) * 255)).astype('uint8')
        neg_img_r = ((neg_img_r - neg_img_r.min())
                          * (1 / (neg_img_r.max() - neg_img_r.min()) * 255)).astype('uint8')


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

        a[3].imshow(img_r,
                       cmap='gray',
                       interpolation='none')
        a[3].axis('off')
        a[4].imshow(a_img_r,
                       cmap='gray',
                       interpolation='none')
        a[4].axis('off')
        a[5].imshow(neg_img_r,
                       cmap='gray',
                       interpolation='none')
        a[5].axis('off')

        img = a[6].imshow(distances_pos_neg[triplet_idx].unsqueeze(dim=1).detach().cpu().numpy(),
                          cmap='gray')#, vmin=0., vmax=1.)
        a[6].axis('off')
        img = a[7].imshow(distances_pos_aug[triplet_idx].unsqueeze(dim=1).detach().cpu().numpy(),
                          cmap='gray')#, vmin=0., vmax=1.)
        a[7].axis('off')
        img = a[8].imshow(distances_aug_neg[triplet_idx].unsqueeze(dim=1).detach().cpu().numpy(),
                          cmap='gray')#, vmin=0., vmax=1.)
        a[8].axis('off')

        img = a[9].imshow(emb_distances_pos[triplet_idx].detach().cpu().numpy(),
                          cmap='gray')#, vmin=0., vmax=1.)
        a[9].axis('off')
        img = a[10].imshow(emb_distances_aug[triplet_idx].detach().cpu().numpy(),
                          cmap='gray')#, vmin=0., vmax=1.)
        a[10].axis('off')
        img = a[11].imshow(emb_distances_neg[triplet_idx].detach().cpu().numpy(),
                          cmap='gray')#, vmin=0., vmax=1.)
        a[11].axis('off')

        if writer:
            img_save_path = os.path.join(config['img_dir'], f'{step:05d}' + f'_train_decoding_result_pair_{triplet_idx}_{batch}.png')
            plt.savefig(img_save_path,
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()

            image = Image.open(img_save_path)
            image = TF.to_tensor(image)
            writer.add_image(f"train_rec/train_decoding_result_pair_{triplet_idx}_{batch}", image, global_step=step)


def plot_train_examples_sanity(model, imgs, writer, config, batch, step=0):
    n_examples_to_show = 2

    z0_recon, z1_recon, z0_recon_swap0, z1_recon_swap0, z0_recon_swap1, z1_recon_swap1 = model.sanity_forward(imgs)

    # specific image pair ids
    for triplet_idx in range(n_examples_to_show):
        f, a = plt.subplots(2, 4)

        a = a.flat

        a[0].set_title('pos', fontsize=10)
        a[1].set_title('recon', fontsize=10)
        a[2].set_title('recon swap 0', fontsize=10)
        a[3].set_title('recon swap 1', fontsize=10)
        a[4].set_title('neg', fontsize=10)
        a[5].set_title('recon', fontsize=10)
        a[6].set_title('recon swap 0', fontsize=10)
        a[7].set_title('recon swap 1', fontsize=10)

        # unnormalize
        img = imgs[0] * 2 + 0.5
        a_img = imgs[1] * 2 + 0.5
        neg_img = imgs[2] * 2 + 0.5

        z0_recon_np = z0_recon[triplet_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()
        z1_recon_np = z1_recon[triplet_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()
        z0_recon_swap0_np = z0_recon_swap0[triplet_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()
        z1_recon_swap0_np = z1_recon_swap0[triplet_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()
        z0_recon_swap1_np = z0_recon_swap1[triplet_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()
        z1_recon_swap1_np = z1_recon_swap1[triplet_idx].permute(1, 2, 0).squeeze().detach().cpu().numpy()

        # convert -1 1 range to 0 255 range for plotting
        z0_recon_np = ((z0_recon_np - z0_recon_np.min())
                          * (1 / (z0_recon_np.max() - z0_recon_np.min()) * 255)).astype('uint8')
        z1_recon_np = ((z1_recon_np - z1_recon_np.min())
                          * (1 / (z1_recon_np.max() - z1_recon_np.min()) * 255)).astype('uint8')
        z0_recon_swap0_np = ((z0_recon_swap0_np - z0_recon_swap0_np.min())
                          * (1 / (z0_recon_swap0_np.max() - z0_recon_swap0_np.min()) * 255)).astype('uint8')
        z1_recon_swap0_np = ((z1_recon_swap0_np - z1_recon_swap0_np.min())
                          * (1 / (z1_recon_swap0_np.max() - z1_recon_swap0_np.min()) * 255)).astype('uint8')
        z0_recon_swap1_np = ((z0_recon_swap1_np - z0_recon_swap1_np.min())
                          * (1 / (z0_recon_swap1_np.max() - z0_recon_swap1_np.min()) * 255)).astype('uint8')
        z1_recon_swap1_np = ((z1_recon_swap1_np - z1_recon_swap1_np.min())
                          * (1 / (z1_recon_swap1_np.max() - z1_recon_swap1_np.min()) * 255)).astype('uint8')


        a[0].imshow(img[triplet_idx].reshape(config['img_shape']).permute(1, 2, 0).squeeze().detach().cpu().numpy(),
                       cmap='gray',
                       interpolation='none')
        a[0].axis('off')
        a[1].imshow(z0_recon_np,
                       cmap='gray',
                       interpolation='none')
        a[1].axis('off')
        a[2].imshow(z0_recon_swap0_np,
                       cmap='gray',
                       interpolation='none')
        a[2].axis('off')
        a[3].imshow(z0_recon_swap1_np,
                       cmap='gray',
                       interpolation='none')
        a[3].axis('off')

        a[4].imshow(neg_img[triplet_idx].reshape(config['img_shape']).permute(1, 2, 0).squeeze().detach().cpu().numpy(),
                       cmap='gray',
                       interpolation='none')
        a[4].axis('off')
        a[5].imshow(z1_recon_np,
                       cmap='gray',
                       interpolation='none')
        a[5].axis('off')
        a[6].imshow(z1_recon_swap0_np,
                       cmap='gray',
                       interpolation='none')
        a[6].axis('off')
        a[7].imshow(z1_recon_swap1_np,
                       cmap='gray',
                       interpolation='none')
        a[7].axis('off')

        if writer:
            img_save_path = os.path.join(config['img_dir'], f'{step:05d}' + f'_train_decoding_result_pair_{triplet_idx}_{batch}_sanity.png')
            plt.savefig(img_save_path,
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()

            image = Image.open(img_save_path)
            image = TF.to_tensor(image)
            writer.add_image(f"train_rec/train_decoding_result_pair_{triplet_idx}_{batch}_sanity", image, global_step=step)


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
