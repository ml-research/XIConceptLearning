import torch
import torchvision
from torchvision import transforms
import os

def get_dataloader(config, data_path="data"):
    dataset = load_data(config, data_path)
    return torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                                       num_workers=config['n_workers'], pin_memory=True)


def load_data(config, data_path):
    if config['dataset'] == 'mnist':
        mnist_data = torchvision.datasets.MNIST(root='dataset', train=True, download=True,
                                                transform=transforms.ToTensor())
        mnist_data_test = torchvision.datasets.MNIST(root='dataset', train=False, download=True,
                                                     transform=transforms.ToTensor())

        config['img_shape'] = (1, 28, 28)
        config['n_prototype_vectors'] = 10
        print('Overriding img_shape and n_prototype_vectors')

        return torch.utils.data.ConcatDataset((mnist_data, mnist_data_test))

    elif config['dataset'] == 'toycolor':
        train_data = np.load(os.path.join(data_path, 'train_toydata.npy'))
        train_data = torch.Tensor(train_data).permute(0, 3, 1, 2)
        train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

        train_labels = np.load(os.path.join(data_path, 'train_toydata_labels.npy'))
        train_labels = torch.Tensor(train_labels)

        config['img_shape'] = (3, 28, 28)
        config['n_prototype_vectors'] = train_labels.shape[1]
        print('Overriding img_shape and n_prototype_vectors')

        return torch.utils.data.TensorDataset(train_data, train_labels)

    elif config['dataset'] == 'toycolorshape':
        train_data = np.load(os.path.join(data_path, 'train_toydata_color_shape.npy'))
        train_data = torch.Tensor(train_data).permute(0, 3, 1, 2)
        train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

        train_labels = np.load(os.path.join(data_path, 'train_toydata_color_shape_labels.npy'))
        train_labels = torch.Tensor(train_labels)

        config['img_shape'] = (3, 28, 28)
        config['n_prototype_vectors'] = train_labels.shape[1]
        print('Overriding img_shape and n_prototype_vectors')

        return torch.utils.data.TensorDataset(train_data, train_labels)

    elif config['dataset'] == 'toycolorshapesize':
        train_data = np.load(os.path.join(data_path, 'train_toydata_color_shape_size.npy'))
        train_data = torch.Tensor(train_data).permute(0, 3, 1, 2)
        train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

        train_labels = np.load(os.path.join(data_path, 'train_toydata_color_shape_size_labels.npy'))
        train_labels = torch.Tensor(train_labels)

        config['img_shape'] = (3, 28, 28)
        config['n_prototype_vectors'] = train_labels.shape[1] + 3  # sizes
        print('Overriding img_shape and n_prototype_vectors')

        return torch.utils.data.TensorDataset(train_data, train_labels)

    else:
        print('Select valid dataset please: mnist, toycolor, toycolorshape, toycolorshapesize')
        exit(42)


"""
Created on Mon Jul 10 21:22:58 2017

Source: https://github.com/OscarcarLi/PrototypeDL
"""
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def batch_elastic_transform(images, sigma, alpha, height, width, random_state=None):
    '''
    this code is borrowed from chsasank on GitHubGist
    Elastic deformation of images as described in [Simard 2003].

    images: a two-dimensional numpy array; we can think of it as a list of flattened images
    sigma: the real-valued variance of the gaussian kernel
    alpha: a real-value that is multiplied onto the displacement fields

    returns: an elastically distorted image of the same shape
    '''
    assert len(images.shape) == 2
    # the two lines below ensure we do not alter the array images
    e_images = np.empty_like(images)
    e_images[:] = images

    e_images = e_images.reshape(-1, height, width)

    if random_state is None:
        random_state = np.random.RandomState(None)
    x, y = np.mgrid[0:height, 0:width]

    for i in range(e_images.shape[0]):
        dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
        dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
        indices = x + dx, y + dy
        e_images[i] = map_coordinates(e_images[i], indices, order=1)

    return e_images.reshape(-1, height * width)
