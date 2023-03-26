import os
import argparse
import torch
import json
import datetime
import numpy as np
from utils import set_seed, makedirs


# parse train options
def _get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda', help='device to be used')
    parser.add_argument('--device-ids', type=int, nargs='+', default=[0], help='device to be used')

    # parser.add_argument('--learn', type=str, required=True, help='unsup, weakly or sup')

    parser.add_argument('--save-step', type=int, default=50, help='save model every # steps')
    parser.add_argument('--print-step', type=int, default=10, help='print metrics every # steps')
    parser.add_argument('--display-step', type=int, default=1,
                        help='track metrics and iamges on tensorboard every # steps')

    # parser.add_argument('--lambda-recon-z', type=float, default=0., help='lambda for z recon loss')
    parser.add_argument('--lambda-recon-proto', type=float, default=1.,
                        help='lambda for agg prototype recon loss')
    parser.add_argument('--gamma-discriminator', type=float, default=1.,
                        help='gamma for discriminator loss')
    parser.add_argument('--gamma-classifier', type=float, default=1.,
                        help='gamma for classifier loss')
    parser.add_argument('--gamma-slots', type=float, default=1.,
                        help='gamma for slot loss')
    parser.add_argument('--lambda-rr', type=float, default=1.,
                        help='lambda for rigth reason mse loss')
    parser.add_argument('--train-protos', action='store_true', default=False, help='should the prototype embedding weights be updated '
                                                                    'too?')
    parser.add_argument('--freeze-enc', action='store_true', help='should the encoder be further finetuned or not')

    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr-scheduler', action='store_true', help='use learning rate scheduler')
    parser.add_argument('--lr-scheduler-warmup-steps', type=int, default=0,
                        help='learning rate scheduler warmup steps')
    parser.add_argument('-bs', '--batch-size', type=int, default=1000, help='batch size, for paired training this is '
                                                                           'the batch size of pairs, so in the end you '
                                                                           'have 2xbs')
    parser.add_argument('-e', '--epochs', type=int, default=3000, help='epochs')
    parser.add_argument('--n-workers', type=int, default=2, help='workers to load data')
    parser.add_argument('--pretrain-encoder', type=int, default=0, help='number of epochs to pretrain encoder.')

    parser.add_argument('-pv', '--prototype-vectors', type=int , nargs='+', default=[4, 4, 2],
                        help='List of number of prototype vectors per category [#p1, #p2, ...]')
    parser.add_argument('--n-protos', type=int, default=2,
                        help='number of classes per categorical variable, i.e. number of prototypes per group')
    parser.add_argument('--proto-dim', type=int, default=32, help='dimensions of each prototype')
    parser.add_argument('--extra-mlp-dim', type=int, default=1, help='dimensions of extra mlp')
    parser.add_argument('--multiheads', action='store_true', default=True,
                        help='should multiple mlp heads be used each for one category??')

    parser.add_argument('--lin-enc-size', type=int, default=512, help='latent dimensions')
    parser.add_argument('--temperature', type=float, default=1., help='temperature of gumbel softmax')
    parser.add_argument('--hack', action='store_true', help='use the original softmax temperature hack')

    parser.add_argument('--exp-name', type=str, default='', help='experiment name')
    parser.add_argument('--results-dir', type=str, default='results', help='results directory')
    parser.add_argument('--model-dir', type=str, default='states', help='model directory')
    parser.add_argument('--img-dir', type=str, default='imgs', help='image\plot directory')
    parser.add_argument('-dd', '--data-dir', type=str, default='Data', help='data directory')

    parser.add_argument('-s', '--seed', type=int, default=42, help='seed')

    parser.add_argument('-d', '--dataset', type=str, default='ecr',
                        help="'toycolor' or 'mnist' or 'toycolorshape' or 'toyshapesize' or 'toycolorshapesize'")
    parser.add_argument('--initials', type=str, default="ML", #required=True,
                        help="Your initials")
    parser.add_argument('--fpath-load-pretrained', type=str, default=None,
                        help='specify the fpath to the pretrained model')
    parser.add_argument('--ckpt-fp', type=str, default=None,
                        help='file path to ckpt file from which to continue training')
    parser.add_argument('--test', action='store_true', help='should we just test?')
    parser.add_argument('--wrong-protos', type=int, nargs='+', action='append', default=None,
                        help='List of list of prototypes to ignore, e.g. 0, 2 for the first group. Please repeat the '
                             'args for each further group, e.g. --wrong-protos 0 2 --wrong-protos 2 4 for a final '
                             'list[[0,2], [2,4]]' )
    parser.add_argument('--pent-id', type=int, default=None, help='which prototype should encode the pentagon shape?')
    parser.add_argument('--circle-id', type=int, default=None, help='which prototype should encode the circle shape?')
    parser.add_argument('--n-per-group', type=int, default=1000, help='how many epochs to learn per prototype group?')
    parser.add_argument('--wandb', action='store_true',
                        help='should results be logged to wandb???')
    return parser


def parse_args(argv):
    """Parses arguments passed from the console as, e.g.
    'python ptt/main.py --epochs 3' """

    parser = _get_parser()
    args = parser.parse_args(argv)

    #args.prototype_vectors = [int(n) for n in args.prototype_vectors[0].split(',')]
    # args.n_prototype_groups = len(args.prototype_vectors)
    if args.exp_name == '':
        args.exp_name = 'seed' + str(args.seed) + '_' \
                        + 'protos' + str(args.prototype_vectors) + '_' + \
                        str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]

    args.results_dir = os.path.join(args.results_dir, args.exp_name)
    args.model_dir = os.path.join(args.results_dir, args.model_dir)
    args.img_dir = os.path.join(args.results_dir, args.img_dir)
    makedirs(args.model_dir)
    makedirs(args.img_dir)

    # set seed for all random processes
    set_seed(args.seed)

    args.device = str(
        args.device + ':' + str(args.device_ids[0]) if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    device_name = str(torch.cuda.get_device_name(args.device) if args.device == "cuda" else args.device)
    print('Device name: {}'.format(device_name))

    args.n_groups = len(args.prototype_vectors)

    # turn list of list into one list with cumulated ids, e.g. [[0, 2], [1, 3]] --> [0, 2, 7, 9] for two groups
    # of 6 prototype vectors
    if args.wrong_protos is not None:
        tmp = []
        pv_cumsum = np.insert(np.cumsum(args.prototype_vectors), 0, 0)
        for i in range(len(args.prototype_vectors)):
            tmp.extend(list((args.wrong_protos[i] + pv_cumsum[i]).astype(int)))
        args.wrong_protos = [int(elem) for elem in tmp]

    with open(os.path.join(args.results_dir, 'args.json'), 'w') as json_file:
        json.dump(vars(args), json_file, indent=2)

    return args


def parse_args_as_dict(argv):
    """Parses arguments passed from the console and returns a dictionary """
    return vars(parse_args(argv))


def parse_dict_as_args(dictionary):
    """Parses arguments given in a dictionary form"""
    argv = []
    for key, value in dictionary.items():
        if isinstance(value, bool):
            if value:
                argv.append('--' + key)
        else:
            argv.append('--' + key)
            argv.append(str(value))
    return parse_args(argv)
