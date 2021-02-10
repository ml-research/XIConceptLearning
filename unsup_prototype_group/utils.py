import scipy.optimize
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from torchvision import transforms
import argparse
from datetime import datetime


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_args(args, writer):
    """
    Create a txt file in the tensorboard writer directory containing all args.
    :param args:
    :param writer:
    :return:
    """
    # store args as txt file
    with open(os.path.join(writer.log_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"\n{arg}: {getattr(args, arg)}")


def write_imgs(writer, epoch, imgs, tag, num_imgs=10):
    """
    Add the reconstructed and original image to tensorboard writer.
    :param writer:
    :param epoch:
    :param imgs:
    :param tag:
    :return:
    """
    for j in range(num_imgs):
        ax = plt.subplot(2, num_imgs // 2, j + 1)
        img = imgs[j].squeeze().detach().cpu()
        plt.imshow(np.array(transforms.ToPILImage()(img).convert("RGB")))
        ax.axis('off')

    fig = plt.gcf()
    writer.add_figure(f"Val_Sample/{tag}", fig, epoch, close=True)

    plt.close()


def write_switch_prototypes(writer, epoch, imgs, recon_protos, recon_imgs, switched_rec_proto):
    """
    Plot the switched reconstructed prototypes, where the attribute of two prototypes has been switched.
    :param writer:
    :param epoch:
    :param imgs:
    :param recon_protos:
    :param recon_imgs:
    :param switched_rec_proto:
    :return:
    """

    for j in range(10):
        fig = plt.figure()
        ax = plt.axes()
        img = imgs[j].squeeze().detach().cpu()
        ax.imshow(np.array(transforms.ToPILImage()(img).convert("RGB")))
        writer.add_figure(f"Val_Switched/Img{j}", fig, epoch, close=True)

        fig = plt.figure()
        ax = plt.axes()
        img = recon_imgs[j].squeeze().detach().cpu()
        ax.imshow(np.array(transforms.ToPILImage()(img).convert("RGB")))
        writer.add_figure(f"Val_Switched/Recon_Img{j}", fig, epoch, close=True)

        fig = plt.figure()
        ax = plt.axes()
        img = recon_protos[j].squeeze().detach().cpu()
        ax.imshow(np.array(transforms.ToPILImage()(img).convert("RGB")))
        writer.add_figure(f"Val_Switched/Recon_Proto{j}", fig, epoch, close=True)

        fig = plt.figure()
        ax = plt.axes()
        img = switched_rec_proto[j].squeeze().detach().cpu()
        ax.imshow(np.array(transforms.ToPILImage()(img).convert("RGB")))
        writer.add_figure(f"Val_Switched/Switched_Recon_Proto{j}", fig, epoch, close=True)

        plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )
    parser.add_argument(
        "--resume", help="Path to log file to resume from"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--warm-up-steps", type=int, default=10, help="Number of steps fpr learning rate scheduler"
    )
    parser.add_argument(
        "--test-log", type=int, default=10, help="Number of epochs before logging AP"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=32, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of threads for data loader"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument(
        "--device-list-parallel", nargs="+", default=None,
        help="List of gpu devices for parallel computing, e.g. None or 1,3"
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )
    parser.add_argument(
        "-d", "--data-dir", type=str, help="Directory to data"
    )

    parser.add_argument(
        "-recx", "--lambda-recon-imgs", type=float, default=1., help="lambda for image reconstruction loss term"
    )
    parser.add_argument(
        "-recp", "--lambda-recon-protos", type=float, default=1., help="lambda for prototype reconstruction loss term"
    )
    parser.add_argument(
        "-r1", "--lambda-r1", type=float, default=1., help="lambda for r1 loss term"
    )
    parser.add_argument(
        "-r2", "--lambda-r2", type=float, default=1., help="lambda for r1 loss term"
    )
    parser.add_argument(
        "-ad", "--lambda-ad", type=float, default=1.,
        help="lambda for attribute decorrelation loss term (from Xu et al. 2020)"
    )
    parser.add_argument(
        "--lambda-enc-mse", type=float, default=1., help="lambda for mse between encodings"
    )

    parser.add_argument(
        "--img-shape", nargs="+", default=None,
        help="List of img shape dims [channel, width, height]"
    )
    parser.add_argument(
        "--n-prototype-groups", type=int, default=2, help="Number of different prototype groups"
    )
    parser.add_argument(
        "--n-prototype-vectors-per-group", type=int, default=3,
        help="Number of prototypes per group (constant over all)"
    )

    parser.add_argument(
        "--seed", type=int, default=3,
        help="Random number seed"
    )

    args = parser.parse_args()

    args.img_shape = np.array([int(dim) for dim in args.img_shape[0].split(',')])

    if args.device_list_parallel is not None:
        args.device_list_parallel = [int(elem) for elem in args.device_list_parallel[0].split(',')]

    # set all seeds for reproducibility
    set_seed(args.seed)

    if not args.no_cuda:
        args.device = "cuda:0"
    else:
        args.device = "cpu"

    return args
