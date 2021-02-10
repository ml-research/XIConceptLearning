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
import model_v1_a as model
import utils as utils
import autoencoder_helpers as ae_helpers
from utils import get_args
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
torch.set_num_threads(6)




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
    cnt = 0
    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):
        if not args.no_cuda:
            imgs, _ = map(lambda x: x.to(args.device), sample)
        else:
            imgs, _ = sample

        std = (args.epochs*iters_per_epoch - i) / args.epochs*iters_per_epoch
        recon_imgs, recon_protos, protos_latent, imgs_latent, per_group_prototype = net.forward(imgs, noise_std=std)

        # TODO: something fishy here with backward
        # R1 Loss: draws prototype close to training example
        tmp = torch.ones(net.n_prototype_groups, requires_grad=False, device=args.device)
        for k in range(net.n_prototype_groups):
            tmp[k] = torch.mean(
                torch.min(ae_helpers.list_of_distances(per_group_prototype[:, k, :].view(-1, net.input_dim_prototype),
                                                       imgs_latent.view(-1, net.input_dim_prototype)), dim=1)[0]
                                    )
        loss_r1 = torch.mean(tmp)

        # add attribute decorrelation loss from Xu et al. 2020 (equation 5)
        loss_ad = torch.sum(torch.sqrt(torch.sum(protos_latent.T**2, dim=1)), dim=0)
        # loss_ad = torch.tensor(0., requires_grad=True)

        # additional mse between two encodings, z_i and z'_i
        loss_enc_mse = criterion(protos_latent, imgs_latent.view(-1, net.input_dim_prototype))

        # compute reconstruction loss between reconstructed prototypes and images
        loss_softmin_proto_recon = criterion(recon_protos, imgs)

        # compute reconstruction loss between reconstructed images and images
        loss_img_recon = criterion(recon_imgs, imgs)

        #loss = args.lambda_recon_imgs * loss_img_recon + \
        #       args.lambda_recon_protos * loss_softmin_proto_recon + \
        #       args.lambda_r1 * loss_r1 + \
        #       args.lambda_ad * loss_ad + \
        #       args.lambda_enc_mse * loss_enc_mse

        loss = args.lambda_recon_protos * loss_softmin_proto_recon + \
               args.lambda_r1 * loss_r1
        # loss = loss_ad

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
                writer.add_scalar("metric/train/loss_ad", loss_ad.item(), global_step=i)
                writer.add_scalar("metric/train/loss_enc_mse", loss_enc_mse.item(), global_step=i)

                print(f"Epoch {epoch} Global Step {i} Train Loss: {loss.item():.6f} |"
                      f"Recon Imgs: {loss_img_recon.item():.6f} |"
                      f"Recon Proto: {loss_softmin_proto_recon.item():.6f} |"
                      f"R1 Loss: {loss_r1.item():.6f} |"
                      f"AD Loss: {loss_ad.item():.6f} |"
                      f"MSE zt' Loss: {loss_enc_mse.item():.6f} ")

            cur_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", cur_lr, global_step=i)
            if args.resume is None:
                # perform scheduler step if warm up is over
                if i >= args.warm_up_steps:
                    scheduler.step()
        else:
            if i == epoch * iters_per_epoch:
                protos = net.get_prototypes()

                with torch.no_grad():
                    for g, protos_group in enumerate(protos):
                        dec_protos = net.dec_prototype(protos_group)
                        utils.write_imgs(writer, i, dec_protos, f'Prototypes{g}', num_imgs=len(dec_protos))

                utils.write_imgs(writer, i, imgs, 'Img_Orig', num_imgs=len(imgs))
                utils.write_imgs(writer, i, recon_protos, 'Rec_with_prototypes', num_imgs=len(recon_protos))
                utils.write_imgs(writer, i, recon_imgs, 'Imgs_Recon', num_imgs=len(imgs))

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
            batch_size=len(dataset_val),
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
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=2e-5)

    # Create RTPT object
    rtpt = RTPT(name_initials='WS', experiment_name=f"ToyData Group Prototypes V1A",
                max_iterations=args.epochs)

    # store args as txt file
    utils.save_args(args, writer)

    # Start the RTPT tracking
    rtpt.start()

    epoch_range = np.arange(start_epoch, args.epochs + start_epoch)
    for epoch in tqdm(epoch_range):
        run_single_epoch(net, train_loader, optimizer, criterion, scheduler, writer, args,
            train=True, epoch=epoch)
        if (epoch + 1) % args.test_log == 0:
            run_single_epoch(net, val_loader, optimizer, criterion, scheduler, writer, args,
                train=False, epoch=epoch)
        rtpt.step()

        if (epoch + 1) % args.test_log == 0 or epoch == 0 or epoch == len(epoch_range) - 1:
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
