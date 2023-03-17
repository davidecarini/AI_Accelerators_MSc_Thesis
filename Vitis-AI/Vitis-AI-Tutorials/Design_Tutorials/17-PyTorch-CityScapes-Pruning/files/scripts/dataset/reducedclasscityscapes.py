# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import os
import sys
import random
import numpy as np
from tqdm import tqdm, trange
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform

from code.datasets.base import BaseDataset

class ReducedCitySegmentation(BaseDataset):
    def __init__(self, root, split='val', mode='testval', transform=None, target_transform=None, **kwargs):
        super(ReducedCitySegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        self.images, self.mask_paths = get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n")
        self._indices = np.array(range(-1, 6))
        self._classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 
                                  9, 10, 11, 12, 13, 17, 24, 
                                 25, 26, 27, 28, 32, 33])
        self._key = np.array([-1,  0,  0,  0,  0,  0,
                               0,  0,  0,  0,  0,  0, 
                               1,  1,  1,  1,  1,  1,
                               1,  1, -1, -1, -1,  0,
                               2,  3,  3,  4,  4,  4,
                              -1, -1, -1,  5, 5])
        self._mapping = np.array(range(-1, len(self._key)-1)).astype('int32')

    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _preprocess(self, mask_file):
        if os.path.exists(mask_file):
            masks = torch.load(mask_file)
            return masks
        masks = []
        print("Preprocessing mask, this will take a while." + \
            "But don't worry, it only run once for each split.")
        tbar = tqdm(self.mask_paths)
        for fname in tbar:
            tbar.set_description("Preprocessing masks {}".format(fname))
            mask = Image.fromarray(self._class_to_index(
                np.array(Image.open(fname))).astype('int8'))
            masks.append(mask)
        torch.save(masks, mask_file)
        return masks

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index])
        if self.mode == 'testval':
            img, mask = self._testval_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_transform(img, mask)
        elif self.mode == 'train':
            img, mask = self._train_transform(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    def make_pred(self, mask):
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._indices)
        index = np.digitize(mask.ravel(), self._indices, right=True)
        return self._classes[index].reshape(mask.shape)


def get_city_pairs(folder, split='val'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []  
        mask_paths = []  
        for root, directories, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit','gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    img_folder = os.path.join(folder, 'leftImg8bit/' + split)
    mask_folder = os.path.join(folder, 'gtFine/'+ split)
    img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    return img_paths, mask_paths
