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


class CLEVR(Dataset):
    def __init__(self, root, mode):
        # path = os.path.join(root, mode)
        self.root = root
        self.mode = mode
        assert os.path.exists(root), 'Path {} does not exist'.format(root)
        
        self.img_paths = []
        for file in os.scandir(os.path.join(root, 'images', mode)):
            img_path = file.path
            self.img_paths.append(img_path)

        self.img_paths.sort()
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = io.imread(img_path)[:, :, :3]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(250),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        img = transform(img)
        img = (img - 0.5) * 2.0  # Rescale to [-1, 1].

        # Google created a tfrecords file with segmentation masks, however I could not convert this to pytorch
        # mask_transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.CenterCrop(192),
        #     transforms.Resize(128, interpolation=Image.NEAREST),
        # ])
        # filename = os.path.split(img_path)[-1]
        # mask_path = os.path.join(self.root, 'masks', filename)
        # mask = None
        # if os.path.exists(mask_path):
        #     mask = io.imread(mask_path)
        #     mask = self.sep(mask)
        #     mask = [np.array(mask_transform(x[:, :, None].astype(np.uint8))) for x in mask]
        #     mask = np.stack(mask, axis=0)
        #     mask = torch.from_numpy(mask.astype(np.float)).float()

        return img, torch.tensor([])#, mask
        
    def __len__(self):
        return len(self.img_paths)

    def sep(self, img):
        """
        Seperate a color image into masks
        :img: (H, W, 3)
        :return: (K, H, W), bool array
        """
        img = img[:, :, :3]
        # a = set()
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         pixel = tuple(img[i, j][:3])
        #         if pixel not in a and pixel != (64, 64, 64):
        #             a.add(pixel)
        H, W, _ = img.shape
        pixels = list(tuple(pix) for pix in img.reshape(H * W, 3))
        a = set(pixels)
        # background
        a.remove((64, 64, 64))
        masks = []
        for pixel in a:
            pixel = np.array(pixel)
            # (h, w, 3)
            mask = img == pixel
            mask = np.all(mask, 2)
            masks.append(mask)
    
        return masks
    
    
    
