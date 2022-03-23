import os
import torch
import torchvision
import pickle
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset


def get_dataloader(config):
    dataset = load_data(config)
    return torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'],
                                       shuffle=True, num_workers=config['n_workers'], pin_memory=True)

def load_data(config):
    # dataloader setup
    if config['dataset'] == 'ecr':
        dataset = ECR_PairswithTest(config['data_dir'], attrs='')
        config['img_shape'] = (3, 64, 64)
        return dataset
    elif config['dataset'] == 'ecr_spot':
        dataset = ECR_PairswithTest(config['data_dir'],
                                            attrs='_spot')
        config['img_shape'] = (3, 64, 64)
        return dataset
    elif config['dataset'] == 'ecr_nospot':
        dataset = ECR_PairswithTest(config['data_dir'],
                                            attrs='_nospot')
        config['img_shape'] = (3, 64, 64)
        return dataset
    else:
        raise ValueError('Select valid dataset please')


def get_test_set(data_loader, config):

    x = data_loader.dataset.test_data
    y = data_loader.dataset.test_labels.tolist()
    y = [sample[0] for sample in y]
    y_set = np.unique(y, axis=0).tolist()
    x_set = []
    for u in y_set:
        x_set.append(np.moveaxis(x[y.index(u)][0], (0, 1, 2), (1, 2, 0)))
    x_set = torch.Tensor(x_set)
    x_set = x_set.to(config['device'])

    y_set = torch.tensor(y_set)
    return x_set, y_set


class ECR_PairswithTest(Dataset):
    def __init__(self, root, attrs, mode='train', single=False):
        self.root = root
        self.single_imgs = single
        assert os.path.exists(root), 'Path {} does not exist'.format(root)

        print("Loading " + os.path.sep.join([root, f"train_ecr{attrs}_pairs.npy"]))

        self.train_data_path = os.path.sep.join([root, mode, f"{mode}_ecr{attrs}_pairs.npy"])
        self.test_data_path = os.path.sep.join([root, "test", f"test_ecr{attrs}_pairs.npy"])
        self.train_labels_path = os.path.sep.join([root, mode, f"{mode}_ecr{attrs}_labels_pairs.pkl"])
        self.test_labels_path = os.path.sep.join([root, "test", f"test_ecr{attrs}_labels_pairs.pkl"])

        self.train_data = np.load(self.train_data_path, allow_pickle=True)
        self.test_data = np.load(self.test_data_path, allow_pickle=True)

        self.train_data = (self.train_data - self.train_data.min()) / (self.train_data.max() - self.train_data.min())
        self.test_data = (self.test_data - self.test_data.min()) / (self.test_data.max() - self.test_data.min())

        with open(self.train_labels_path, 'rb') as f:
            labels_dict = pickle.load(f)
            self.train_labels = labels_dict['labels_one_hot']
            self.train_labels_as_id = labels_dict['labels']
            try:
                self.shared_labels = labels_dict['shared_labels']
            except:
                self.shared_labels = None
        with open(self.test_labels_path, 'rb') as f:
            labels_dict = pickle.load(f)
            self.test_labels = labels_dict['labels_one_hot']

    def __getitem__(self, index):
        imgs = self.train_data[index]

        labels_one_hot = self.train_labels[index]
        labels_ids = self.train_labels_as_id[index]

        # compute which category is shared unless it was precomputed
        if self.shared_labels is not None:
            shared_labels = self.shared_labels[index].astype(np.bool)
        else:
            shared_labels = (labels_ids[0] == labels_ids[1])

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        # transform the positive negative samples
        img0 = transform(np.uint8(imgs[0]*255)).float()
        img1 = transform(np.uint8(imgs[1]*255)).float()
        # img_size = tuple(img0.shape[-2:])

        if self.single_imgs:
            return torch.cat((img0, img1), dim=0), torch.cat((labels_one_hot[0], labels_one_hot[1]), dim=0), \
                   torch.cat((labels_ids[0], labels_ids[1]), dim=0)
        else:
            return (img0, img1), (labels_one_hot[0], labels_one_hot[1]), (labels_ids[0], labels_ids[1]), shared_labels

    def __len__(self):
        return len(self.train_data)


