import os
import random
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing as mp

import scipy.optimize
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from captum.attr import InputXGradient, IntegratedGradients, DeepLift, NoiseTunnel
from captum.attr._core.layer.grad_cam import LayerGradCam

import data as data
import model
import utils as utils
from rtpt import RTPT

torch.autograd.set_detect_anomaly(True)

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
torch.set_num_threads(6)

# -----------------------------------------
# - Define basic and data related methods -
# -----------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )
    parser.add_argument("--mode", type=str, required=True, help="train, test, or plot")
    parser.add_argument("--resume", help="Path to log file to resume from")

    parser.add_argument(
        "--seed", type=int, default=10, help="Random generator seed for all frameworks"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--l2_grads", type=float, default=1, help="Right for right reason weight"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of threads for data loader"
    )
    parser.add_argument(
        "--dataset",
        choices=["clevr-hans-state"],
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs")

    parser.add_argument("--data-dir", type=str, help="Directory to data")
    parser.add_argument("--fp-ckpt", type=str, default=None, help="checkpoint filepath")

    # Slot attention params
    parser.add_argument('--n-slots', default=10, type=int,
                        help='number of slots for slot attention module')
    parser.add_argument('--n-iters-slot-att', default=3, type=int,
                        help='number of iterations in slot attention module')
    parser.add_argument('--n-attr', default=18, type=int,
                        help='number of attributes per object')

    args = parser.parse_args()

    # hard set !!!!!!!!!!!!!!!!!!!!!!!!!
    args.n_heads = 4
    args.set_transf_hidden = 128

    assert args.data_dir.endswith(os.path.sep)
    args.conf_version = args.data_dir.split(os.path.sep)[-2]
    args.name = args.name + f"-{args.conf_version}"

    if args.mode == 'test' or args.mode == 'plot':
        assert args.fp_ckpt

    if args.no_cuda:
        args.device = 'cpu'
    else:
        args.device = 'cuda'

    seed_everything(args.seed)

    return args


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------
# - Define Train/Test/Validation methods -
# -----------------------------------------


def train(args):

    if args.dataset == "clevr-hans-state":
        dataset_train = data.CLEVR(
            args.data_dir, "train"
        )
        dataset_test = data.CLEVR(
            args.data_dir, "test"
        )
    else:
        print("Wrong dataset specifier")
        exit()

    args.n_imgclasses = dataset_train.n_classes
    args.class_weights = torch.ones(args.n_imgclasses)/args.n_imgclasses
    args.classes = np.arange(args.n_imgclasses)
    args.category_ids = dataset_train.category_ids

    train_loader = data.get_loader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    test_loader = data.get_loader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    val_loader = data.get_loader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # net = model.IMG2TabCls(args, n_slots=args.n_slots, n_iters=args.n_iters_slot_att, n_attr=args.n_attr,
    #                        set_transf_hidden=args.set_transf_hidden, category_ids=args.category_ids,
    #                        device=args.device)
    #
    # # load pretrained state predictor
    # log = torch.load("logs/slot-attention-clevr-state-3_final")
    # net.img2state_net.load_state_dict(log['weights'], strict=True)
    # print("Pretrained slot attention model loaded!")
    #
    # net = net.to(args.device)
    #
    # # only optimize the set transformer classifier for now, i.e. freeze the state predictor
    # optimizer = torch.optim.Adam(
    #     [p for name, p in net.named_parameters() if p.requires_grad and 'set_cls' in name], lr=args.lr
    # )
    # criterion = nn.CrossEntropyLoss()
    # criterion_lexi = LexiLoss(class_weights=args.class_weights.float().to("cuda"), args=args)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)
    #
    # torch.backends.cudnn.benchmark = True
    #
    # # Create RTPT object
    # rtpt = RTPT(name_initials='WS', experiment_name=f"Clevr Hans Slot Att Set Transf xil",
    #             max_iterations=args.epochs)
    # # Start the RTPT tracking
    # rtpt.start()
    #
    # # tensorboard writer
    # writer = utils.create_writer(args)
    # # writer = None
    #
    # cur_best_val_loss = np.inf
    # for epoch in range(args.epochs):
    #     _ = run_lexi(net, train_loader, optimizer, criterion, criterion_lexi, split='train', args=args,
    #                          writer=writer, train=True, plot=False, epoch=epoch)
    #     scheduler.step()
    #     val_loss = run_lexi(net, val_loader, optimizer, criterion, criterion_lexi, split='val', args=args,
    #                         writer=writer, train=False, plot=True, epoch=epoch)
    #     _ = run(net, test_loader, optimizer, criterion, split='test', args=args, writer=writer,
    #             train=False, plot=False, epoch=epoch)
    #
    #     results = {
    #         "name": args.name,
    #         "weights": net.state_dict(),
    #         "args": args,
    #     }
    #     if cur_best_val_loss > val_loss:
    #         if epoch > 0:
    #             # remove previous best model
    #             os.remove(glob.glob(os.path.join(writer.log_dir, "model_*_bestvalloss*.pth"))[0])
    #         torch.save(results, os.path.join(writer.log_dir, "model_epoch{}_bestvalloss_{:.4f}.pth".format(epoch,
    #                                                                                                        val_loss)))
    #         cur_best_val_loss = val_loss
    #
    #     # Update the RTPT (subtitle is optional)
    #     rtpt.step()
    #
    # # load best model for final evaluation
    # net = model.IMG2TabCls(args, n_slots=args.n_slots, n_iters=args.n_iters_slot_att, n_attr=args.n_attr,
    #                        set_transf_hidden=args.set_transf_hidden, category_ids=args.category_ids,
    #                        device=args.device)
    # net = net.to(args.device)
    # checkpoint = torch.load(glob.glob(os.path.join(writer.log_dir, "model_*_bestvalloss*.pth"))[0])
    # net.load_state_dict(checkpoint['weights'])
    # net.eval()
    # print("\nModel loaded from checkpoint for final evaluation\n")
    #
    # get_confusion_from_ckpt(net, test_loader, criterion, args=args, datasplit='test_best',
    #                         writer=writer)
    # get_confusion_from_ckpt(net, val_loader, criterion, args=args, datasplit='val_best',
    #                         writer=writer)
    #
    # # plot expls
    # run(net, train_loader, optimizer, criterion, split='train_best', args=args, writer=writer, train=False, plot=True, epoch=0)
    # run(net, val_loader, optimizer, criterion, split='val_best', args=args, writer=writer, train=False, plot=True, epoch=0)
    # run(net, test_loader, optimizer, criterion, split='test_best', args=args, writer=writer, train=False, plot=True, epoch=0)
    #
    # writer.close()


def main():
    args = get_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        pass
    elif args.mode == 'plot':
        pass


if __name__ == "__main__":
    main()
