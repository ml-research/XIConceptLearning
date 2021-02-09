import os
import argparse
import random
import numpy as np
from datetime import datetime
import time

import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from model import CAE
from autoencoder_helpers import makedirs # TODO: move functions to utils
from network import *  # TODO: move functions to utils 
import data as data

from data_preprocessing import batch_elastic_transform

from hessian_penalty_pytorch import hessian_penalty

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rtpt.rtpt import RTPT

def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )

    parser.add_argument("--resume", help="Path to log file to resume from")
    
    parser.add_argument(
        "--seed", type=int, default=42, help="Random generator seed for all frameworks"
    )

    # training
    parser.add_argument(
        "--epochs", type=int, default=5000, help="# of epochs to train with"
    )
    # preset lr works best
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=250, help="Batch size to train with"
    )
    parser.add_argument(
        "--save-step", type=int, default=500, help="Save model every # steps"
    )
    parser.add_argument(
        "--test-display-step", type=int, default=10, help="Write tensorboard and test model every # steps"
    )
    parser.add_argument(
        "--results-dir", type=str, default='results', help="Directory of results"
    )

    # hardware
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument("--device-id", default=0, type=int, help="ID of GPU device")
    
    parser.add_argument(
        "--n-workers", type=int, default=4, help="# of threads for data loader"
    )
    
    # dataset
    #TODO: remove data-dir default
    parser.add_argument("--data-dir", default='../../datasets/no_conf_1obj/', type=str, help="Directory to data")
    parser.add_argument("--class-select", default='all', type=str, help="Classes to train on (coords, shape, size, material, color, all)")

    parser.add_argument("--img-dim-h", default=128, type=int, help="Image dimensions h")
    parser.add_argument("--img-dim-w", default=128, type=int, help="Image dimensions w")
    parser.add_argument("--img-dim-c", default=3, type=int, help="Image channels c")

    # prototype layer params
    parser.add_argument('--n-prototype-vectors', default=10, type=int,
                        help='# of prototype vectors for prototype layer, -1 for n_classes+1')
    parser.add_argument('--n-z', default=10, type=int,
                        help='latent dimension of latent representation/prototype vectors')
    
    # regularization
    parser.add_argument("--batch-elastic-transform", action="store_true", help="Use batch elastic transform")

    # loss
    #TODO: paper proposed 20-1-1-1-0 but using CLEVR dataset works best for 10-10-1-1-0 as reconstruction is harder
    parser.add_argument("--lambda-class", default=10, type=float, help="lambda class loss")
    parser.add_argument("--lambda-ae", default=10, type=float, help="lambda autoencoder loss")
    parser.add_argument("--lambda-r1", default=1, type=float, help="lambda R1 loss")
    parser.add_argument("--lambda-r2", default=1, type=float, help="lambda R2 loss")
    #TODO: hessian penalty doesn't make improvements (maybe apply to different layers and not to the whole network...) 
    parser.add_argument("--lambda-hessian", default=0, type=float, help="lambda hessian penalty")

    args = parser.parse_args()

    if args.no_cuda:
        args.device = 'cpu'
    else:
        args.device = f'cuda:{args.device_id}'

    class_dict = dict(
        {
            '?': list(range(0,1)),
            'coords': list(range(1,4)),
            'shape': list(range(4,7)),
            'size': list(range(7,9)),
            'material': list(range(9,11)),
            'color': list(range(11,18)),
            'all': list(range(0, 18))
        }
    )

    args.idx_classes = class_dict[args.class_select]
    args.n_classes = len(args.idx_classes)

    if args.n_prototype_vectors == -1:
        print('Using n_prototype_vectors = n_classes+1')
        args.n_prototype_vectors = args.n_classes + 1

    seed_everything(args.seed)

    return args

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    # create directories
    results_dir = os.path.join(args.results_dir, args.name)
    
    tfb_dir = os.path.join(results_dir, args.name)
    makedirs(tfb_dir)
    model_dir=os.path.join(results_dir, 'states')
    makedirs(model_dir)
    img_dir = os.path.join(results_dir, 'imgs')
    makedirs(img_dir)

    writer = create_writer(tfb_dir, args)

    # create datasets and dataloader
    clevr_data = data.CLEVR(args.data_dir, 'train', resize=(args.img_dim_h, args.img_dim_w))
    data_loader = data.get_loader(
            clevr_data,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=True,
        )

    clevr_data_test = data.CLEVR(args.data_dir, 'test', resize=(args.img_dim_h, args.img_dim_w))
    data_loader_test = data.get_loader(
            clevr_data_test,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=True,
        )

    # model setup
    model = CAE(input_dim=(1,args.img_dim_c, args.img_dim_h, args.img_dim_w), n_z=args.n_z, n_prototype_vectors=args.n_prototype_vectors, n_classes=args.n_classes).to(args.device)

    # optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # create RTPT object
    rtpt = RTPT(name_initials='MM', experiment_name='XIC_PrototypeDL', max_iterations=args.epochs)
    rtpt.start()

    for e in range(0, args.epochs):
        train_acc = 0
        max_iter = len(data_loader)
        start = time.time()

        class_loss = 0
        r1_loss = 0 
        r2_loss = 0 
        ae_loss = 0
        hessian_loss = 0

        for i, batch in enumerate(data_loader):
            imgs = batch[0]
            # to apply elastic transform, batch has to be flattened
            # store original size to put it back into orginial shape after transformation
            if args.batch_elastic_transform:
                imgs_shape = imgs.shape
                imgs = batch_elastic_transform(imgs.view(args.batch_size, -1), sigma=args.sigma, alpha=args.alpha, height=args.img_dim_h, width=args.img_dim_w)
                imgs = torch.reshape(torch.tensor(imgs), imgs_shape)
            imgs = imgs.to(args.device)

            labels = batch[1][:,:,args.idx_classes].squeeze().to(args.device)

            optimizer.zero_grad()        

            pred = model.forward(imgs)

            # class loss
            
            if args.class_select in ['color', 'shape', 'size', 'material']:
                _, labels_idx = labels.max(1)
                criterion = torch.nn.CrossEntropyLoss()
                class_loss_batch = criterion(pred, labels_idx)

            elif args.class_select in ['coords']:
                mse = torch.nn.MSELoss()
                class_loss_batch = mse(pred, labels)

            else: # all
                softmax = torch.nn.Softmax(dim=1)        
                huber_loss = torch.nn.SmoothL1Loss()
                class_loss_batch = huber_loss(softmax(pred), softmax(labels))


            prototype_vectors = model.prototype_layer.prototype_vectors
            feature_vectors = model.feature_vectors

            # R1 & R2 regularizations
            r1_loss_batch = torch.mean(torch.min(list_of_distances(prototype_vectors, feature_vectors.view(-1, model.input_dim_prototype)), dim=1)[0])
            r2_loss_batch = torch.mean(torch.min(list_of_distances(feature_vectors.view(-1, model.input_dim_prototype ), prototype_vectors), dim=1)[0])
            
            # reconstruct images
            rec = model.forward_dec(feature_vectors)

            # autoencoder loss (paper & mse)
            ae_loss_batch = torch.mean(list_of_norms(rec-imgs))
            
            # mse = torch.nn.MSELoss()
            # ae_loss = mse(rec, imgs)

            # hessian penalty
            hessian_loss_batch = hessian_penalty(G=model, z=imgs)

            loss = args.lambda_class * class_loss_batch +\
                    args.lambda_r1 * r1_loss_batch +\
                    args.lambda_r2 * r2_loss_batch +\
                    args.lambda_ae * ae_loss_batch +\
                    args.lambda_hessian * hessian_loss_batch

            loss.backward()
            
            optimizer.step()

            class_loss += class_loss_batch
            r1_loss += r1_loss_batch
            r2_loss += r2_loss_batch
            ae_loss += ae_loss_batch
            hessian_loss += hessian_loss_batch

            # train accuracy
        #     max_vals, max_indices = torch.max(pred,1)
        #     n = max_indices.size(0)
        #     train_acc += (max_indices == labels).sum(dtype=torch.float32)/n
        
        # train_acc /= max_iter
        
        class_loss /= max_iter
        r1_loss /= max_iter
        r2_loss /= max_iter
        ae_loss /= max_iter
        hessian_loss /= max_iter
        
        # debug
        train_acc = torch.tensor(train_acc)
        # print general info
        print(f'epoch {e} - time {round(time.time()-start,2)} - loss {round(loss.item(),6)} - train accuracy {round(train_acc.item(),6)}')
        # print losses
        print('class', round(class_loss.item(),4), 'r1', round(r1_loss.item(),4), 'r2', round(r2_loss.item(),4), 'ae_loss', round(ae_loss.item(),4), 'hessian_penalty', round(hessian_loss.item(), 4))

        # RTPT step
        rtpt.step(subtitle=f"loss={loss.item():2.2f}")

        if (e+1) % args.save_step == 0 or e == args.epochs - 1:
            
            # save model states
            state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ep': e
                    }
            torch.save(state, os.path.join(model_dir, '%05d.pth' % (e)))

            # save results as images
            dec_in_dim =  model.dec_out_shapes[-1][0] // 2
        
            # decode prototype vectors
            prototype_imgs = model.forward_dec(prototype_vectors.reshape((-1,args.n_z,dec_in_dim,dec_in_dim))).detach().cpu()
            prototype_imgs = prototype_imgs.permute(0,2,3,1)
            
            # visualize the prototype images
            n_cols = 5
            if args.n_prototype_vectors < n_cols:
                n_cols = args.n_prototype_vectors // 2 + 1

            n_rows = args.n_prototype_vectors // n_cols + 1 if args.n_prototype_vectors % n_cols != 0 else args.n_prototype_vectors // n_cols
            g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
            for i in range(n_rows):
                for j in range(n_cols):
                    if i*n_cols + j < args.n_prototype_vectors:
                        if args.img_dim_c == 1:
                            b[i][j].imshow(prototype_imgs[i*n_cols + j].reshape(args.img_dim_h, args.img_dim_w),
                                            cmap='gray',
                                            interpolation='none')
                            b[i][j].axis('off')
                        else:
                            b[i][j].imshow(prototype_imgs[i*n_cols + j].reshape(args.img_dim_h, args.img_dim_w, args.img_dim_c),
                                        # cmap='gray',
                                        interpolation='none')
                            b[i][j].axis('off')

            # print(prototype_imgs.shape)
            
            grid = torchvision.utils.make_grid(prototype_imgs.permute(0,3,1,2), n_cols)
            writer.add_image('Image', grid)
            # torchvision.utils.save_image(grid, os.path.join(img_dir, 'prototype_result-' + str(e) + '.png'))
            # writer.close()
            # exit(42) 
            plt.savefig(os.path.join(img_dir, 'prototype_result-' + str(e) + '.png'),
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()

            # apply encoding and decoding over a small subset of the training set
            imgs = []
            for batch in data_loader:
                imgs = batch[0].to(args.device)
                break

            examples_to_show = 10
            
            encoded = model.enc.forward(imgs[:examples_to_show])
            decoded = model.dec.forward(encoded)

            decoded = decoded.detach().cpu()
            decoded = decoded.permute(0,2,3,1)
            imgs = imgs.detach().cpu()
            imgs = imgs.permute(0,2,3,1)

            # compare original images to their reconstructions
            f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))

            if args.img_dim_c == 1:
                for i in range(examples_to_show):
                    a[0][i].imshow(imgs[i].reshape(args.img_dim_h, args.img_dim_w),
                                    cmap='gray',
                                    interpolation='none')
                    a[0][i].axis('off')
                    a[1][i].imshow(decoded[i].reshape(args.img_dim_h, args.img_dim_w), 
                                    cmap='gray',
                                    interpolation='none')
                    a[1][i].axis('off')
            else:
                for i in range(examples_to_show):
                    a[0][i].imshow(imgs[i].reshape(args.img_dim_h, args.img_dim_w, args.img_dim_c),
                                    interpolation='none')
                    a[0][i].axis('off')
                    a[1][i].imshow(decoded[i].reshape(args.img_dim_h, args.img_dim_w, args.img_dim_c),
                                    interpolation='none')
                    a[1][i].axis('off')

            # writer.add_figure('figure/recon', f, global_step=e, close=False)

            plt.savefig(os.path.join(img_dir, 'decoding_result-' + str(e) + '.png'),
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()

            print(f'SAVED - epoch {e} - imgs @ {img_dir} - model @ {model_dir}')

        # test set accuracy evaluation
        if (e+1) % args.test_display_step == 0 or e == args.epochs - 1:
            # max_iter = len(data_loader_test)
            # test_acc = 0

            # for i, batch in enumerate(data_loader_test):
            #     imgs = batch[0]
            #     imgs = imgs.to(args.device)
            #     labels = batch[1].to(args.device)

            #     pred = model.forward(imgs)

            #     # test accuracy
            # #     max_vals, max_indices = torch.max(pred,1)
            # #     n = max_indices.size(0)
            # #     test_acc += (max_indices == labels).sum(dtype=torch.float32)/n

            # # test_acc /= max_iter

            # # debug
            # test_acc = torch.tensor(test_acc)
            # print(f'TEST - epoch {e} - accuracy {round(test_acc.item(),6)}')

            # write to tensorboard
            writer.add_scalar('loss/class', class_loss.item(), e)
            writer.add_scalar('loss/ae', ae_loss.item(), e)
            writer.add_scalar('loss/r1', r1_loss.item(), e)
            writer.add_scalar('loss/r2', r2_loss.item(), e)
            writer.add_scalar('loss/hessian', hessian_loss.item(), e)

def create_writer(tfb_dir, args):
    writer = SummaryWriter(tfb_dir, purge_step=0)

    writer.add_scalar('hyperparameters/learningrate', args.lr, 0)
    writer.add_scalar('hyperparameters/num_epochs', args.epochs, 0)
    writer.add_scalar('hyperparameters/batchsize', args.batch_size, 0)

    writer.add_scalar('hyperparameters/lambdaClass', args.lambda_class, 0)
    writer.add_scalar('hyperparameters/lambdaAE', args.lambda_ae, 0)
    writer.add_scalar('hyperparameters/lambdaR1', args.lambda_r1, 0)
    writer.add_scalar('hyperparameters/lambdaR2', args.lambda_r1, 0)
    writer.add_scalar('hyperparameters/lambdaHessian', args.lambda_hessian, 0)

    # store args as txt file
    with open(os.path.join(tfb_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"\n{arg}: {getattr(args, arg)}")
    return writer

def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()