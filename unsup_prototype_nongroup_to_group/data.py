import os
import torch
import torchvision
from torchvision import transforms
import numpy as np

def get_dataloader(config):
    dataset = load_data(config)
    return torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['n_workers'], pin_memory=True)

def load_data(config):
    # dataloader setup
    if config['dataset'] == 'mnist':
        mnist_data = torchvision.datasets.MNIST(root='dataset', train=True, download=True,
                                                transform=transforms.ToTensor())
        mnist_data_test = torchvision.datasets.MNIST(root='dataset', train=False, download=True,
                                                     transform=transforms.ToTensor())

        config['img_shape'] = (1, 28, 28)
        config['prototype_vectors'] = [10]
        print('Overriding img_shape and n_prototype_vectors')

        return torch.utils.data.ConcatDataset((mnist_data, mnist_data_test))

    elif config['dataset'] == 'toycolor':
        train_data = np.load(os.path.join(config['data_dir'],'train_toydata_color.npy'))
        train_data = torch.Tensor(train_data).permute(0, 3, 1, 2)
        train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

        train_labels = np.load(os.path.join(config['data_dir'],'train_toydata_color_labels.npy'))
        train_labels = torch.Tensor(train_labels)

        config['img_shape'] = (3, 28, 28)
        # config['prototype_vectors'] = train_labels.shape[1]
        print('Overriding img_shape')

        return torch.utils.data.TensorDataset(train_data, train_labels)

    elif config['dataset'] == 'toycolorshape':
        train_data = np.load(os.path.join(config['data_dir'],'train_toydata_color_shape.npy'))
        train_data = torch.Tensor(train_data).permute(0, 3, 1, 2)
        train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

        train_labels = np.load(os.path.join(config['data_dir'],'train_toydata_color_shape_labels.npy'))
        train_labels = torch.Tensor(train_labels)

        config['img_shape'] = (3, 28, 28)
        print('Overriding img_shape and n_prototype_vectors')

        return torch.utils.data.TensorDataset(train_data, train_labels)

    elif config['dataset'] == 'toyshapesize':
        train_data = np.load(os.path.join(config['data_dir'],'train_toydata_shape_size.npy'))
        train_data = torch.Tensor(train_data).permute(0,3,1,2)
        train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

        # print(train_data)
        train_labels = np.load(os.path.join(config['data_dir'],'train_toydata_shape_size_labels.npy'))
        train_labels = torch.Tensor(train_labels)

        config['img_shape'] = (3,28,28)
        config['prototype_vectors'] = 6
        print('Overriding img_shape and n_prototype_vectors')

        return torch.utils.data.TensorDataset(train_data, train_labels)

    elif config['dataset'] == 'toycolorshapesize':
        train_data = np.load(os.path.join(config['data_dir'],'train_toydata_color_shape_size.npy'))
        train_data = torch.Tensor(train_data).permute(0,3,1,2)
        train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

        train_labels = np.load(os.path.join(config['data_dir'],'train_toydata_color_shape_size_labels.npy'))
        train_labels = torch.Tensor(train_labels)

        config['img_shape'] = (3,28,28)
        config['prototype_vectors'] = train_labels.shape[1] + 3 # sizes
        print('Overriding img_shape and n_prototype_vectors')

        return torch.utils.data.TensorDataset(train_data, train_labels)

    else:
        raise ValueError('Select valid dataset please: mnist, toycolor')
