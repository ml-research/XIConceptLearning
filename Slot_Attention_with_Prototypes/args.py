import os
import random
import numpy as np
import torch
import argparse
from datetime import datetime

from utils.utils import hungarian_loss

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
        "--epochs", type=int, default=10000, help="# of epochs to train with"
    )
    # preset lr works best
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--warm-up-steps", type=int, default=1000, help="# of steps for learning rate scheduler"
    )
    parser.add_argument(
        "--lr-policy", type=str, default='lambda', help="learning rate policy"
    )
    parser.add_argument(
        "--epochs-decay", type=int, default=5000, help="epoch start decay learning rate, set -1 if no decay"
    )
    parser.add_argument(
        "--batch-size", type=int, default=250, help="Batch size to train with"
    )
    parser.add_argument(
        "--save-step", type=int, default=500, help="Save model every # steps"
    )
    parser.add_argument(
        "--examples-to-show", type=int, default=10, help="# examples to be plotted"
    )
    parser.add_argument(
        "--display-step", type=int, default=10, help="Write tensorboard and test model every # steps"
    )
    parser.add_argument(
        "--results-dir", type=str, default='results', help="Directory of results"
    )

    # hardware
    parser.add_argument("--no-cuda", action="store_true", help="Run on CPU instead of GPU (not recommended)",)
    parser.add_argument("--device-ids", nargs="+", default=[0], type=int, help="ID(s) of GPU device(s)")
    
    parser.add_argument("--n-workers", type=int, default=4, help="# of threads for data loader * # of GPUs")
    
    # dataset
    #TODO: remove data-dir default
    parser.add_argument("--data-dir", default='../../datasets/no_conf_1obj/', type=str, help="Directory to data")
    parser.add_argument("--class-select", default='all', type=str, help="Classes to train on (coords, shape, size, material, color, all)")

    parser.add_argument("--img-dim-h", default=128, type=int, help="Image dimensions h")
    parser.add_argument("--img-dim-w", default=128, type=int, help="Image dimensions w")
    parser.add_argument("--img-dim-c", default=3, type=int, help="Image channels c")

    # prototype layer params
    parser.add_argument("--n-prototype-vectors", default=-1, type=int, help="# of prototype vectors for prototype layer, -1 for n_classes+1")
    parser.add_argument("--n-slots", default=11, type=int, help="number of slots for slot attention, 11 for pretrained")
    parser.add_argument("--freeze", action="store_true", help="freeze encoder and decoder, set R2 and reconstruction loss to 0")
    
    # regularization
    parser.add_argument("--batch-elastic-transform", action="store_true", help="Use batch elastic transform")

    # loss
    #TODO: paper proposed 20-1-1-1-0 but using CLEVR dataset works best for 10-10-1-1-0 as reconstruction is harder
    parser.add_argument("--lambda-class", default=10, type=float, help="lambda class loss")
    parser.add_argument("--lambda-ae", default=10, type=float, help="lambda autoencoder loss")
    parser.add_argument("--lambda-r1", default=1, type=float, help="lambda R1 loss")
    parser.add_argument("--lambda-r2", default=1, type=float, help="lambda R2 loss")
    parser.add_argument("--lambda-hung", default=10, type=float, help="lambda hungarian loss")
    parser.add_argument("--lambda-recon", default=1, type=float, help="lambda reconstruction loss")
    #TODO: hessian penalty doesn't make improvements (maybe apply to different layers and not to the whole network...) 
    parser.add_argument("--lambda-hessian", default=0, type=float, help="lambda hessian penalty")

    args = parser.parse_args()

    if args.no_cuda:
        args.device = 'cpu'
    else:
        args.device = f'cuda:{args.device_ids[0]}'

    args.n_workers = len(args.device_ids) * args.n_workers

    args.class_dict = dict(
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

    args.class_losses = dict(
        {
            '?': None,
            'coords': torch.nn.MSELoss,
            'shape': torch.nn.BCEWithLogitsLoss,
            'size': torch.nn.BCEWithLogitsLoss,
            'material': torch.nn.BCEWithLogitsLoss,
            'color': torch.nn.BCEWithLogitsLoss,
            'all': hungarian_loss
        }
    )

    args.idx_classes = args.class_dict[args.class_select]
    args.n_classes = len(args.idx_classes)

    print('Overriding class_select')
    args.classes_select = ['color', 'shape']
    args.idx_classes = []
    for class_select in args.classes_select:
        args.idx_classes += list(args.class_dict[class_select])
        print(f'selected class {class_select} of shape {len(list(args.class_dict[class_select]))}')
    
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