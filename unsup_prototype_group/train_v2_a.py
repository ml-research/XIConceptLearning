import os
import argparse
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
from torch.optim import lr_scheduler
from datetime import datetime
from rtpt import RTPT

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import data as data
# import model as model
import model_v2_a as model
import utils as utils
import autoencoder_helpers as ae_helpers

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
torch.set_num_threads(6)

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

    parser.add_argument(
        "--lambda-recon-imgs", type=float, default=1., help="lambda for image reconstruction loss term"
    )
    parser.add_argument(
        "--lambda-recon-protos", type=float, default=1., help="lambda for prototype reconstruction loss term"
    )
    parser.add_argument(
        "--lambda-r1", type=float, default=1., help="lambda for r1 loss term"
    )
    parser.add_argument(
        "--lambda-r2", type=float, default=1., help="lambda for r1 loss term"
    )
    parser.add_argument(
        "--lambda-ad", type=float, default=1., help="lambda for attribute decorrelation loss term (from Xu et al. 2020)"
    )

    parser.add_argument(
        "--img-shape", nargs="+", default=None,
        help="List of img shape dims [batch, width, height]"
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
    utils.set_seed(args.seed)

    if not args.no_cuda:
        args.device = "cuda:0"
    else:
        args.device = "cpu"

    return args


def run_single_epoch(net, loader, optimizer, criterion, scheduler, writer, args, train=False, epoch=0):
    if train:
        net.train()
        prefix = "train"
        torch.set_grad_enabled(True)
    else:
        net.eval()
        prefix = "test"
        torch.set_grad_enabled(False)

    iters_per_epoch = len(loader)

    for i, sample in tqdm(enumerate(loader, start=epoch * iters_per_epoch)):
        if not args.no_cuda:
            imgs, labels = map(lambda x: x.to("cuda:0"), sample)
        else:
            imgs, labels = sample

        recon_imgs, recon_protos, protos_latent, imgs_latent = net.forward(imgs)

        loss_r1 = torch.mean(
            torch.min(ae_helpers.list_of_distances(protos_latent.view(-1, net.input_dim_prototype),
                                                   imgs_latent.view(-1, net.input_dim_prototype)),
                      dim=1
                      )[0]
        )

        # draws encoding close to prototype
        loss_r2 = torch.mean(
            torch.min(ae_helpers.list_of_distances(imgs_latent.view(-1, net.input_dim_prototype),
                                                   protos_latent.view(-1, net.input_dim_prototype)),
                      dim=1
                      )[0]
        )

        # add attribute decorrelation loss from Xu et al. 2020 (equation 5)
        loss_ad = torch.sum(torch.sqrt(torch.sum(protos_latent.T**2, dim=1)), dim=0)

        # compute reconstruction loss between reconstructed prototypes and images
        loss_softmin_proto_recon = criterion(recon_protos, imgs)

        # compute reconstruction loss between reconstructed images and images
        loss_img_recon = criterion(recon_imgs, imgs)

        # loss = args.lambda_recon_imgs * loss_img_recon + \
        #        args.lambda_recon_protos * loss_softmin_proto_recon + \
        #        args.lambda_r1 * loss_r1 + \
        #        args.lambda_r2 * loss_r2 + \
        #        args.lambda_ad * loss_ad
        # loss = args.lambda_recon_imgs * loss_img_recon + \
        #        args.lambda_recon_protos * loss_softmin_proto_recon + \
        loss = args.lambda_recon_protos * loss_softmin_proto_recon
               # args.lambda_recon_imgs * loss_img_recon

        if train:

            if args.resume is None:
                # manual lr warmup
                if i < args.warm_up_steps:
                    learning_rate = args.lr * (i+1)/args.warm_up_steps
                    optimizer.param_groups[0]["lr"] = learning_rate

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                writer.add_scalar("metric/train/loss", loss.item(), global_step=i)
                writer.add_scalar("metric/train/loss_recon_imgs", loss_img_recon.item(), global_step=i)
                writer.add_scalar("metric/train/loss_softmin_proto_recon", loss_softmin_proto_recon.item(),
                                  global_step=i)
                writer.add_scalar("metric/train/loss_r1", loss_r1.item(), global_step=i)
                # writer.add_scalar("metric/train/loss_r2", loss_r2.item(), global_step=i)
                writer.add_scalar("metric/train/loss_ad", loss_ad.item(), global_step=i)

                print(f"Epoch {epoch} Global Step {i} Train Loss: {loss.item():.6f} "
                      f"Recon Imgs: {loss_img_recon.item():.6f} Recon Proto: {loss_softmin_proto_recon.item():.6f} "
                      f"R1 Loss: {loss_r1.item():.6f} AD Loss: {loss_ad.item():.6f}")

            cur_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", cur_lr, global_step=i)
            if args.resume is None:
                # perform scheduler step if warm up is over
                if i >= args.warm_up_steps:
                    scheduler.step()
        else:

            if i % (iters_per_epoch * args.test_log) == 0:
                utils.write_imgs(writer, i, imgs, 'Img_Orig')
                utils.write_imgs(writer, i, recon_protos, 'Protos_Recon')
                utils.write_imgs(writer, i, recon_imgs, 'Imgs_Recon')

                # # plot mixing prototypes
                # # switch the prototype of one group between two samples
                # tmp = torch.clone(per_group_prototype)
                # per_group_prototype[0, 0, :] = tmp[1, 0, :]
                # per_group_prototype[1, 0, :] = tmp[0, 0, :]
                # # compute the mixture of group prototypes to a single prototype
                # switched_proto_latent = net.mix_prototype_per_sample(per_group_prototype,
                #                                                      batch_size=imgs.shape[0])
                # # reconstruct the prototype
                # switched_rec_proto = net.dec_prototype(switched_proto_latent, imgs_latent.shape)
                # utils.write_switch_prototypes(writer, i, imgs, recon_protos, recon_imgs, switched_rec_proto)

            writer.add_scalar("metric/val/loss", loss.item(), global_step=i)


def train(args):
    writer = SummaryWriter(f"runs/{args.name}", purge_step=0)

    dataset_train = data.ToyData(
        args.data_dir, "train",
    )
    dataset_val = data.ToyData(
        args.data_dir, "val",
    )

    if not args.eval_only:
        train_loader = data.get_loader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    if not args.train_only:
        val_loader = data.get_loader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    net = model.RAE(input_dim=(1, args.img_shape[0], args.img_shape[1], args.img_shape[2]),
                    n_prototype_groups=args.n_prototype_groups,
                    n_prototype_vectors_per_group=args.n_prototype_vectors_per_group,
                    device=args.device)

    start_epoch = 0
    if args.resume:
        print("Loading ckpt ...")
        log = torch.load(args.resume)
        weights = log["weights"]
        net.load_state_dict(weights, strict=True)
        start_epoch = log["args"]["epochs"]

    net = net.to(args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    num_steps = len(train_loader) * args.epochs - args.warm_up_steps
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0.00005)

    # Create RTPT object
    rtpt = RTPT(name_initials='WS', experiment_name=f"ToyData Group Prototypes V2A",
                max_iterations=args.epochs)

    # store args as txt file
    utils.save_args(args, writer)

    # Start the RTPT tracking
    rtpt.start()

    for epoch in np.arange(start_epoch, args.epochs + start_epoch):
        run_single_epoch(net, train_loader, optimizer, criterion, scheduler, writer, args,
            train=True, epoch=epoch)
        run_single_epoch(net, val_loader, optimizer, criterion, scheduler, writer, args,
            train=False, epoch=epoch)
        rtpt.step()

        results = {
            "name": args.name,
            "weights": net.state_dict(),
            "args": vars(args),
        }
        print(os.path.join(writer.log_dir, args.name+'.pth'))
        torch.save(results, os.path.join(writer.log_dir, args.name))
        if args.eval_only:
            break


def test(args):
    pass


def main():
    args = get_args()

    if not args.eval_only:
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
