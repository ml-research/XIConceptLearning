import torchvision
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True


def init_dataset(config):
    # dataloader setup
    if config['dataset'] == 'mnist':
        mnist_data = torchvision.datasets.MNIST(root='dataset', train=True, download=True,
                                                transform=transforms.ToTensor())
        mnist_data_test = torchvision.datasets.MNIST(root='dataset', train=False, download=True,
                                                     transform=transforms.ToTensor())

        config['img_shape'] = (1, 28, 28)
        config['n_prototype_vectors'] = [10]
        print('Overriding img_shape and n_prototype_vectors')

        dataset = torch.utils.data.ConcatDataset((mnist_data, mnist_data_test))

    elif config['dataset'] == 'toycolor':
        train_data = np.load('data/train_toydata.npy')
        train_data = torch.Tensor(train_data).permute(0, 3, 1, 2)
        train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

        train_labels = np.load('data/train_toydata_labels.npy')
        train_labels = torch.Tensor(train_labels)

        config['img_shape'] = (3, 28, 28)
        # config['n_prototype_vectors'] = train_labels.shape[1]
        print('Overriding img_shape')

        dataset = torch.utils.data.TensorDataset(train_data, train_labels)

    elif config['dataset'] == 'toycolorshape':
        train_data = np.load('data/train_toydata_color_shape.npy')
        train_data = torch.Tensor(train_data).permute(0, 3, 1, 2)
        train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())

        train_labels = np.load('data/train_toydata_color_shape_labels.npy')
        train_labels = torch.Tensor(train_labels)

        config['img_shape'] = (3, 28, 28)
        print('Overriding img_shape and n_prototype_vectors')

        dataset = torch.utils.data.TensorDataset(train_data, train_labels)

    elif config['dataset'] == 'toycolorpairs':
        dataset = ToyDataPaired(
            'data/', "train",
        )
        config['img_shape'] = (3, 28, 28)

    else:
        raise ValueError('Select valid dataset please: mnist, toycolor')

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                                              num_workers=config['n_workers'], pin_memory=True)

    x_set = []
    if config['dataset'] in ['toycolorshape', 'toycolor']:
        # generate set of all individual samples
        x = data_loader.dataset.tensors[0].detach().numpy().tolist()
        y = data_loader.dataset.tensors[1].detach().numpy().tolist()
        y_set = np.unique(y, axis=0).tolist()
        x_set = []
        for u in y_set:
            x_set.append(x[y.index(u)])
        x_set = torch.Tensor(x_set)
        x_set = x_set.to(config['device'])
    elif config['dataset'] in ['toycolorpairs']:
        x = data_loader.dataset.data.tolist()
        y = data_loader.dataset.labels.tolist()
        # get label of first img in pair
        y_1 = [sample[0] for sample in y]
        y_set = np.unique(y_1, axis=0).tolist()
        x_set = []
        for u in y_set:
            x_set.append(x[y_1.index(u)])
        x_set = torch.Tensor(x_set)
        x_set = x_set.to(config['device'])

    return data_loader, x_set


def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


class ToyDataPaired(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        assert mode in ['train', 'val']
        assert os.path.exists(root), 'Path {} does not exist'.format(root)
        
        self.data_path = os.path.sep.join([root, f"{mode}_toydata_pairs.npy"])
        self.labels_path = os.path.sep.join([root, f"{mode}_toydata_labels_pairs.npy"])

        self.data = np.load(self.data_path, allow_pickle=True)
        # TODO: check if normalisation is required, conflict currently with transform resize
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        # self.data.astype(np.float64)
        self.labels = np.load(self.labels_path, allow_pickle=True)

    def __getitem__(self, index):
        imgs = self.data[index]
        labels = self.labels[index]

        transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        img1 = transform(imgs[0]).float()
        img2 = transform(imgs[1]).float()

        return (img1, img2), (labels[0], labels[1])
        
    def __len__(self):
        return len(self.data)


    
    
