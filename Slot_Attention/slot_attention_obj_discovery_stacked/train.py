import os
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import scipy.optimize
import numpy as np
from tqdm import tqdm
import matplotlib
from torch.optim import lr_scheduler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import data as data
import model as model
import utils as utils
from rtpt import RTPT

torch.set_num_threads(30)

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
        "--epochs", type=int, default=10, help="Number of epochs to train with"
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
        "--batch-size", type=int, default=32, help="Batch size to train with"
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
        "--data-dir", type=str, help="Directory to data"
    )
    # Slot attention params
    parser.add_argument(
        "--n-slots", default=11, type=int,
        help="number of slots for slot attention module"
    )
    parser.add_argument(
        "--n-attr-slots", default=5, type=int,
        help="number of slots for stacked slot attention module"
    )
    parser.add_argument(
        "--n-iters-slot-att", default=3, type=int,
        help="number of iterations in slot attention module"
    )
    parser.add_argument(
        "--n-attr", default=18, type=int,
        help="number of attributes per object"
    )

    args = parser.parse_args()

    if args.device_list_parallel is not None:
        args.device_list_parallel = [int(elem) for elem in args.device_list_parallel[0].split(',')]

    return args


def run(net, loader, optimizer, criterion, scheduler, writer, args, train=False, epoch=0):
    # train only attribute slot attention submodule
    if train:
        net.train()
        net.obj_slot_attention.requires_grad = False
        prefix = "train"
        torch.set_grad_enabled(True)
    else:
        net.eval()
        prefix = "test"
        torch.set_grad_enabled(False)

    iters_per_epoch = len(loader)

    for i, sample in tqdm(enumerate(loader, start=epoch * iters_per_epoch)):
        imgs, masks = map(lambda x: x.to("cuda:0"), sample)

        _, _, _, slots, recon_combined_attr, recons_attr, masks_attr, stacked_slots = \
            net.forward(imgs)

        # TODO: Why is loss 0?
        loss = criterion(slots, recon_combined_attr)

        if train:

            if args.resume is None:
                # manual lr warmup
                if i < args.warm_up_steps:
                    learning_rate = args.lr * (i+1)/args.warm_up_steps
                    optimizer.param_groups[0]["lr"] = learning_rate

            if not i % 100 or i == 0:
                utils.write_attr_recon_imgs_plots(writer, i, recon_combined_attr, slots)
                utils.write_attr_slot_imgs(writer, i, recons_attr)
                utils.write_attr_mask_imgs(writer, i, masks_attr)
                utils.write_attr_slots(writer, i, stacked_slots)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("metric/train_loss", loss.item(), global_step=i)
            print(f"Epoch {epoch} Global Step {i} Train Loss: {loss.item():.6f}")

            cur_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", cur_lr, global_step=i)
            if args.resume is None:
                # normal lr scheduler
                if i >= args.warm_up_steps:
                    scheduler.step()
        else:

            # if i % (iters_per_epoch * args.test_log) == 0:
                # utils.write_recon_imgs_plots(writer, epoch, recon_combined, imgs, i)

            writer.add_scalar("metric/val_loss", loss.item(), global_step=i)


def train(args):
    writer = SummaryWriter(f"runs/{args.name}", purge_step=0)
    # writer = None

    dataset_train = data.CLEVR(
        args.data_dir, "train",
    )

    train_loader = data.get_loader(
        dataset_train, batch_size=args.batch_size, num_workers=args.num_workers
    )

    net = model.SlotAttention_model(n_slots=args.n_slots, n_attr_slots=args.n_attr_slots,
                                    n_iters=args.n_iters_slot_att,
                                    encoder_hidden_channels=64, attr_encoder_hidden_channels=900,
                                    attention_hidden_channels=128, decoder_hidden_channels=64,
                                    decoder_initial_size=(8, 8))

    # load pretrained object detector
    log = torch.load("logs/slot-attention-clevr-objdiscovery-14")
    weights = utils.convert_dataparallel_weights(log["weights"])
    net.obj_slot_attention.load_state_dict(weights, strict=True)

    start_epoch = 0
    if args.resume:
        print("Loading ckpt ...")
        log = torch.load(args.resume)
        weights = log["weights"]
        net.load_state_dict(weights, strict=True)
        start_epoch = log["args"]["epochs"]

    if not args.no_cuda:
        net = net.to("cuda:0")

    # only optimize the attribute discovery architecture
    optimizer = torch.optim.Adam(net.attr_slot_attention.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    num_steps = len(train_loader) * args.epochs - args.warm_up_steps
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0.00005)
    print(f"{num_steps} training steps ...")

    # Create RTPT object
    rtpt = RTPT(name_initials='WS', experiment_name=f"Stacked Slot Att",
                max_iterations=args.epochs)

    # store args as txt file
    utils.save_args(args, writer)

    # Start the RTPT tracking
    rtpt.start()

    for epoch in np.arange(start_epoch, args.epochs + start_epoch):
        run(net, train_loader, optimizer, criterion, scheduler, writer, args,
            train=True, epoch=epoch)
        rtpt.step()

        results = {
            "name": args.name,
            "weights": net.state_dict(),
            "args": vars(args),
        }
        print(os.path.join("logs", args.name))
        torch.save(results, os.path.join("logs", args.name))
        if args.eval_only:
            break


def test(args):
    dataset_test = data.CLEVR(
        args.data_dir, "val",
    )

    test_loader = data.get_loader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    net = model.SlotAttention_model(n_slots=args.n_slots, n_iters=args.n_iters_slot_att, n_attr=args.n_attr,
                                    encoder_hidden_channels=64, attention_hidden_channels=128,
                                    decoder_hidden_channels=64, decoder_initial_size=(8, 8))

    net = torch.nn.DataParallel(net, device_ids=range(len(args.device_list_parallel)))

    start_epoch = 0
    if args.resume:
        print("Loading ckpt ...")
        log = torch.load(args.resume)
        weights = log["weights"]
        net.load_state_dict(weights, strict=True)
        start_epoch = log["args"]["epochs"]


    if not args.no_cuda:
        net = net.to("cuda:0")

    # Create RTPT object
    rtpt = RTPT(name_initials='WS', experiment_name=f"Obj Discovery Slot Att",
                max_iterations=args.epochs)

    # Start the RTPT tracking
    rtpt.start()

    sample = next(iter(test_loader))

    imgs, masks = map(lambda x: x.to("cuda:0"), sample)
    recon_combined, recons, _, slots = net.forward(imgs)

    # get attention masks and plot
    sample_id = 1
    fig, axs = plt.subplots(nrows=3, ncols=4)
    axs = axs.flatten()
    for i in range(args.n_slots):
        axs[i].imshow(net.module.slot_attention.attn[sample_id, i, :].reshape(128, 128).detach().cpu().numpy(), cmap='gray')
    orig_img = np.moveaxis(imgs[sample_id].detach().cpu().numpy(), [0, 1, 2], [2, 0, 1]) / 2. + 0.5
    axs[11].imshow(orig_img)
    fig.savefig('tmp_attn.png')

    rtpt.step()


def main():
    args = get_args()

    if not args.eval_only:
        train(args)
    else:
        test(args)



if __name__ == "__main__":
    main()
