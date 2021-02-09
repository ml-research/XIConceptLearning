from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


class ToyData(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        assert mode in ['train', 'val']
        assert os.path.exists(root), 'Path {} does not exist'.format(root)
        
        self.data_path = os.path.sep.join([root, f"{mode}_toydata.npy"])
        self.labels_path = os.path.sep.join([root, f"{mode}_toydata_labels.npy"])

        self.data = np.load(self.data_path, allow_pickle=True)
        self.labels = np.load(self.labels_path, allow_pickle=True)

        if mode == 'val':
            x = self.data
            y = self.labels.tolist()
            y_set = np.unique(y, axis=0).tolist()

            x_set = []
            for u in y_set:
                x_set.append(x[y.index(u)])
            self.data = np.array(x_set)
            self.labels = np.array(y_set)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        img = transform(img)

        return img, label
        
    def __len__(self):
        return len(self.data)


    
    
