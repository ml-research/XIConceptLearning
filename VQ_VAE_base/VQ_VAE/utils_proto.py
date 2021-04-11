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

    fig, ax = plt.subplots(nrows=int(np.ceil(np.sqrt(len(comb_proto_ids)))),
                           ncols=int(np.ceil(np.sqrt(len(comb_proto_ids)))))
    ax = ax.flatten()

    prototype_img = model.dec_proto_by_selection(comb_proto_ids).detach().cpu()

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
        labels = labels.to(config['device'])
    else:
        imgs = log_samples
    imgs = imgs.to(config['device'])
    examples_to_show = len(imgs)

    if rec_protos is None:
        if config['learn'] == 'sup':
            vq_loss, _, recons_proto, perplexity, embeddings, embedding_ids_one_hot, _ = model.forward(imgs,
                                                                                                       labels=None)
        elif config['learn'] == 'weakly':
            assert type(imgs) is not tuple
            vq_loss, _, recons_proto, perplexity, embeddings, embedding_ids_one_hot, _ = model.forward(imgs)
        else:
            raise ValueError("unsup learning is not currently handled! "
                             "Please contact schramowski@cs.tu-darmstadt.de for help.")
        embedding_ids_one_hot = torch.flatten(embedding_ids_one_hot.detach().cpu().permute(1, 0, 2), start_dim=1)

    recons_proto = recons_proto.detach().cpu()
    imgs = imgs.detach().cpu()

    # compare original images to their reconstructions
    n_rows = 3
    f, a = plt.subplots(n_rows, examples_to_show, figsize=(examples_to_show, n_rows))

    a[0][0].text(0, -2, s='input', fontsize=10)
    a[1][0].text(0,-2, s='recon proto', fontsize=10)
    a[2][0].text(0,-2, s='prediction', fontsize=10)

    for i in range(examples_to_show):
        a[0][i].imshow(imgs[i].reshape(config['img_shape']).permute(1, 2, 0).squeeze(),
                       cmap='gray',
                       interpolation='none')
        a[0][i].axis('off')

        # convert to RGB numpy array
        recons_np = recons_proto[i].reshape(config['img_shape']).permute(1, 2, 0).squeeze().numpy()
        # convert -1 1 range to 0 255 range for plotting
        recons_np = ((recons_np - recons_np.min())
                          * (1 / (recons_np.max() - recons_np.min()) * 255)).astype('uint8')
        a[1][i].imshow(recons_np,
                       cmap='gray',
                       interpolation='none')
        a[1][i].axis('off')

        img = a[2][i].imshow(np.array([labels[i].detach().cpu().numpy(), embedding_ids_one_hot[i].numpy()]),
                       cmap='gray',
                       vmin=0., vmax=1.)
        a[2][i].axis('off')
        # if i == examples_to_show-1:
        #     f.colorbar(img)

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


def plot_train_examples(imgs, recons_proto, distances, writer, config, step=0):
    n_examples_to_show = 2
    n_rows = 3
    n_cols = n_examples_to_show * 2
    f, a = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))

    # specific image pair ids
    for i, pair_idx in enumerate([0, n_examples_to_show]):

        a[0, i*2].set_title('input', fontsize=10)
        a[1, i*2].set_title('recon proto', fontsize=10)
        a[2, i*2].set_title('distances', fontsize=10)

        for img_idx in range(2):
            col_idx = img_idx + i*2
            a[0, col_idx].imshow(imgs[img_idx][pair_idx].reshape(config['img_shape']).permute(1, 2, 0).squeeze().detach().cpu().numpy(),
                           cmap='gray',
                           interpolation='none')
            a[0, col_idx].axis('off')

            # convert to RGB numpy array
            recons_np = recons_proto[img_idx][pair_idx].reshape(config['img_shape']).permute(1, 2, 0).squeeze().detach().cpu().numpy()
            # convert -1 1 range to 0 255 range for plotting
            recons_np = ((recons_np - recons_np.min())
                              * (1 / (recons_np.max() - recons_np.min()) * 255)).astype('uint8')
            a[1, col_idx].imshow(recons_np,
                           cmap='gray',
                           interpolation='none')
            a[1, col_idx].axis('off')

            img = a[2, col_idx].imshow(distances[img_idx][:, pair_idx].detach().cpu().numpy(),
                           cmap='gray',
                           vmin=-1., vmax=1.)
            a[2, col_idx].axis('off')

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
