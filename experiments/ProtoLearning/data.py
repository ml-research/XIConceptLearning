import os
import torch
import pickle
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd

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
    if config['dataset'] == 'cub':
        config['img_shape'] = (3, 244, 244)
        dataset = CUB_PairswithTest(config['data_dir'], config['img_shape'])
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


class CUB_PairswithTest(Dataset):
    def __init__(self, root, image_shape, mode='train', single_imgs=False, test_amount=32, num_attributes=28):
        self.root = root
        self.data_path = Path(self.root) / "CUB_200_2011"
        self.single_imgs = single_imgs

        self.test_amount = test_amount
        self.num_attributes = num_attributes
        self.image_shape = image_shape
        assert os.path.exists(self.data_path), 'Path {} does not exist'.format(self.data_path)

        print(f"Loading {self.data_path}/244x244/CUB_200_2011.pkl")

        self.train_imgs = np.load(self.data_path/"244x244/CUB_200_2011_train.npy")
        self.test_imgs = np.load(self.data_path/"244x244/CUB_200_2011_test.npy")
        self.labels = np.load(self.data_path/"attributes/one_hot_processed_attributes.npy")

        self.attributes, self.train_image_id_to_index, self.train_index_to_image_id, self.test_image_id_to_index, self.test_index_to_image_id = pickle.load( open( self.data_path/"244x244/CUB_200_2011.pkl", "rb" ) )

        self.train_ids = np.array(list(self.train_image_id_to_index))
        self.test_ids = np.array(list(self.test_image_id_to_index))

        self.processed_attributes = pd.read_pickle(self.data_path /"attributes/processed_attributes.pkl")
        self.train_processed_attributes = self.processed_attributes.loc[self.train_ids].iloc[:, :self.num_attributes]
        self.test_processed_attributes = self.processed_attributes.loc[self.test_ids].iloc[:, :self.num_attributes]

        self.attribute_ranges = pickle.load( open( self.data_path/"attributes/attribute_ranges.pkl", "rb" ) )
        self.num_one_hot_attributes = self.attribute_ranges[self.test_processed_attributes.iloc[:, :self.num_attributes].columns[-1]][-1]

        self.train_labels = self.labels[self.train_ids-1]

        self.train_imgs = (self.train_imgs - self.train_imgs.min()) / (self.train_imgs.max() - self.train_imgs.min())
        self.test_imgs = (self.test_imgs - self.test_imgs.min()) / (self.test_imgs.max() - self.test_imgs.min())


    def __getitem__(self, index):
        image_id0 = self.train_index_to_image_id[index]
        img0 = self.train_imgs[index]

        binary_attributes0 = self.labels[image_id0, :self.num_one_hot_attributes]
        attributes0 = self.processed_attributes.loc[image_id0][:self.num_attributes]
        labels_one_hot0 = binary_attributes0.astype(float).tolist()

        # find image with shared attributes
        matching_candidates = (self.train_processed_attributes.iloc[:, :self.num_attributes] == attributes0).any(axis=1)
        possible_swaps = self.train_processed_attributes.loc[matching_candidates]
        possible_swaps = possible_swaps.drop(image_id0)
        image_id1 = possible_swaps.sample(n=1).index[0]
        img1 = self.train_imgs[self.train_image_id_to_index[image_id1]]
        binary_attributes1 = self.labels[image_id1, :self.num_one_hot_attributes]
        attributes1 = self.processed_attributes.loc[image_id1][:self.num_attributes]
        labels_one_hot1 = binary_attributes1.astype(float).tolist()

        # compute which categories are shared
        shared_labels = (attributes0 == attributes1).to_numpy()


        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        # transform the positive negative samples
        img0 = transform(np.uint8(img0*255)).float()
        img1 = transform(np.uint8(img1*255)).float()

        if self.single_imgs:
            return torch.cat((img0, img1), dim=0), torch.cat((labels_one_hot0, labels_one_hot1), dim=0)
        else:
            return (img0, img1), (labels_one_hot0, labels_one_hot1), shared_labels

    @property
    def test_data(self):
        # 32,1,500,500,3 np.array
        return np.expand_dims(self.test_imgs[:self.test_amount], axis=1)

    @property
    def test_labels(self):
        # list[32[2xlist[10x float]]]
        test_labels = self.labels[self.test_ids-1][:self.test_amount,:self.num_one_hot_attributes].astype(float)
        return np.expand_dims(test_labels, axis=1)

    def __len__(self):
        return len(self.train_imgs)