import os
import math
import random
import json

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
import torchvision.transforms.functional as T
import h5py
import numpy as np

from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
torch.set_num_threads(6)

def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


CLASSES = {
    "shape": ["sphere", "cube", "cylinder"],
    "size": ["large", "small"],
    "material": ["rubber", "metal"],
    "color": ["cyan", "blue", "yellow", "purple", "red", "green", "gray", "brown"],
}


class CLEVR(torch.utils.data.Dataset):
    def __init__(self, base_path, split, resize=(128,128)):
        assert split in {
            "train",
            "val",
            "test",
        }
        self.base_path = base_path
        self.split = split
        self.max_objects = 1

        with open(self.scenes_path) as fd:
            scenes = json.load(fd)["scenes"]

        self.img_ids, self.img_class_ids, self.scenes, self.fnames = self.prepare_scenes(scenes)

        self.transform = transforms.Compose(
            [transforms.Resize(resize),
             transforms.ToTensor()]
        )

        self.n_classes = len(np.unique(self.img_class_ids))
        self.category_dict = CLASSES

        # get ids of category ranges, i.e. shape has three categories from ids 0 to 2
        self.category_ids = np.array([3, 6, 8, 10, 18])

    def convert_coords(self, obj, scene_directions):
        # coords = position
        # Originally the x, y, z positions are in [-3, 3].
        # We re-normalize them to [0, 1].
        # coords = (obj["3d_coords"] + 3.) / 6.
        # from slot attention
        # coords = [(p +3.)/ 6. for p in position]
        # convert the 3d coords based on camera position
        # conversion from ns-vqa paper, normalization for slot attention
        position = [np.dot(obj['3d_coords'], scene_directions['right']),
                    np.dot(obj['3d_coords'], scene_directions['front']),
                    obj['3d_coords'][2]]
        coords = [(p +4.)/ 8. for p in position]
        return coords

    def object_to_fv(self, obj, scene_directions):
        coords = self.convert_coords(obj, scene_directions)
        one_hot = lambda key: [obj[key] == x for x in CLASSES[key]]
        material = one_hot("material")
        color = one_hot("color")
        shape = one_hot("shape")
        size = one_hot("size")
        assert sum(material) == 1
        assert sum(color) == 1
        assert sum(shape) == 1
        assert sum(size) == 1
        # concatenate all the classes
        return coords + shape + size + material + color

    def prepare_scenes(self, scenes_json):
        img_ids = []
        scenes = []
        img_class_ids = []
        fnames = []
        for scene in scenes_json:
            fnames.append(os.path.join(self.images_folder, scene['image_filename']))
            img_class_ids.append(scene['class_id'])
            img_idx = scene["image_index"]

            objects = [self.object_to_fv(obj, scene['directions']) for obj in scene["objects"]]
            objects = torch.FloatTensor(objects).transpose(0, 1)

            num_objects = objects.size(1)
            # pad with 0s
            if num_objects < self.max_objects:
                objects = torch.cat(
                    [
                        objects,
                        torch.zeros(objects.size(0), self.max_objects - num_objects),
                    ],
                    dim=1,
                )

            # fill in masks
            mask = torch.zeros(self.max_objects)
            mask[:num_objects] = 1

            # concatenate obj indication to end of object list
            objects = torch.cat((mask.unsqueeze(dim=0), objects), dim=0)

            img_ids.append(img_idx)
            scenes.append(objects.T)
            # scenes.append(torch.cat(scene['objects']['shape'], scence['objects']['color'])) #material
        return img_ids, img_class_ids, scenes, fnames

    @property
    def images_folder(self):
        return os.path.join(self.base_path, self.split, "images")

    @property
    def scenes_path(self):
        return os.path.join(
            self.base_path, self.split, "CLEVR_HANS_scenes_{}.json".format(self.split)
        )

    def __getitem__(self, item):
        image_id = self.img_ids[item]

        image = pil_loader(self.fnames[item])

        if self.transform is not None:
            image = self.transform(image) # in range [0., 1.]
            # causes images to be dark
            # image = (image - 0.5) * 2.0  # Rescale to [-1, 1].

        objects = self.scenes[item]
        img_class_id = self.img_class_ids[item]

        # remove objects presence indicator from gt table
        objects = objects[:, 1:]

        return image, objects, img_class_id, image_id

    def __len__(self):
        return len(self.scenes)

