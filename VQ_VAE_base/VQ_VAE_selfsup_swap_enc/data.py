import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset


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

        config['img_shape'] = (3, 64, 64)
        print('Overriding img_shape and n_prototype_vectors')

        return torch.utils.data.TensorDataset(train_data, train_labels)

    # elif config['dataset'] == 'toyshapesize':
    #     train_data = np.load(os.path.join(config['data_dir'],'train_toydata_shape_size.npy'))
    #     train_data = torch.Tensor(train_data).permute(0,3,1,2)
    #     train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
    #
    #     # print(train_data)
    #     train_labels = np.load(os.path.join(config['data_dir'],'train_toydata_shape_size_labels.npy'))
    #     train_labels = torch.Tensor(train_labels)
    #
    #     config['img_shape'] = (3,28,28)
    #     config['prototype_vectors'] = 6
    #     print('Overriding img_shape and n_prototype_vectors')
    #
    #     return torch.utils.data.TensorDataset(train_data, train_labels)

    # elif config['dataset'] == 'toycolorshapesize':
    #     train_data = np.load(os.path.join(config['data_dir'],'train_toydata_color_shape_size.npy'))
    #     train_data = torch.Tensor(train_data).permute(0,3,1,2)
    #     train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
    #
    #     train_labels = np.load(os.path.join(config['data_dir'],'train_toydata_color_shape_size_labels.npy'))
    #     train_labels = torch.Tensor(train_labels)
    #
    #     config['img_shape'] = (3,28,28)
    #     config['prototype_vectors'] = train_labels.shape[1] + 3 # sizes
    #     print('Overriding img_shape and n_prototype_vectors')
    #
    #     return torch.utils.data.TensorDataset(train_data, train_labels)

    elif config['dataset'] == 'toycolorshapepairs':
        dataset = ToyDataTriplet(
            config['data_dir'], "train", attrs='color_shape'
        )
        config['img_shape'] = (3, 64, 64)

        return dataset

    else:
        raise ValueError('Select valid dataset please: mnist, toycolor')


def get_test_set(data_loader, config):
    x_set = []
    if config['dataset'] in ['toycolorshape', 'toycolor', 'toycolorshapesize']:
        # generate set of all individual samples
        x = data_loader.dataset.tensors[0].detach().numpy().tolist()
        y = data_loader.dataset.tensors[1].detach().numpy().tolist()
        y_set = np.unique(y, axis=0).tolist()
        x_set = []
        for u in y_set:
            x_set.append(x[y.index(u)])
        x_set = torch.Tensor(x_set)
        x_set = x_set.to(config['device'])
    elif 'pairs' in config['dataset']:

        x = data_loader.dataset.data
        y = data_loader.dataset.labels.tolist()
        # get label of first img in pair
        y_1 = [sample[0] for sample in y]
        y_set = np.unique(y_1, axis=0).tolist()
        x_set = []
        for u in y_set:
            x_set.append(np.moveaxis(x[y_1.index(u)][0], (0, 1, 2), (1, 2, 0)))
        x_set = torch.Tensor(x_set)
        x_set = x_set.to(config['device'])

    y_set = torch.tensor(y_set)
    return x_set, y_set


def get_augmentation_transforms(s=1.0, size=(64, 64)):
    """
    s is the strength of color distortion.
    Code from Chen et al. 2020 (https://arxiv.org/pdf/2002.05709.pdf)
    """
    color_jitter = transforms.ColorJitter(0.9*s, 0.9*s, 0.9*s, 0.1*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    # rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        transforms.ToPILImage(),
        rnd_color_jitter,
        # rnd_gray,
        transforms.RandomRotation(degrees=15, fill=(128, 128, 128)),
        transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    return color_distort


class ToyDataTriplet(Dataset):
    def __init__(self, root, mode, attrs):
        self.root = root
        assert mode in ['train', 'val']
        assert os.path.exists(root), 'Path {} does not exist'.format(root)

        self.data_path = os.path.sep.join([root, f"{mode}_toydata_{attrs}_pairs.npy"])
        self.labels_path = os.path.sep.join([root, f"{mode}_toydata_{attrs}_labels_pairs.npy"])

        self.data = np.load(self.data_path, allow_pickle=True)
        # TODO: check if normalisation is required, conflict currently with transform resize
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        # self.data.astype(np.float64)
        # order of labels is [RECTANGLE, CIRCLE, CYAN, RED, YELLOW, GREEN]
        self.labels = np.load(self.labels_path, allow_pickle=True)

    def __getitem__(self, index):
        imgs = self.data[index]
        labels = self.labels[index]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((64, 64)),
            transforms.RandomRotation(degrees=15, fill=(128, 128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

        # transform the positive negative samples
        img0 = transform(np.uint8(imgs[0]*255)).float()
        img1 = transform(np.uint8(imgs[1]*255)).float()
        img_size = tuple(img0.shape[-2:])

        # augment the positive sample by randomly adding color jitter and cropping
        aug_transform = get_augmentation_transforms(s=0.25, size=img_size)
        aug_img0 = aug_transform(np.uint8(imgs[0]*255)).float()

        # import matplotlib.pyplot as plt
        # fig1, ax = plt.subplots(3, 1)
        # ax[0].imshow(img0.permute(1, 2, 0).detach().cpu().numpy())
        # ax[1].imshow(aug_img0.permute(1, 2, 0).detach().cpu().numpy())
        # ax[2].imshow(img1.permute(1, 2, 0).detach().cpu().numpy())
        # fig1.savefig('tmp.png')

        # convert 0 to 1 to -1 to 1
        img0 = (img0 - 0.5) / 2
        aug_img0 = (aug_img0 - 0.5) / 2
        img1 = (img1 - 0.5) / 2

        return (img0, aug_img0, img1), (labels[0], labels[0], labels[1])

    def __len__(self):
        return len(self.data)
