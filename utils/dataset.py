# Copyright (c) 2024 Ming-Yang Ho, Min-Sheng Wu
# All rights reserved.
#
# This source code is licensed under the AGPL License found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.util import get_transforms


def remove_file(files, file_name):
    try:
        files.remove(file_name)
    except Exception:
        pass


class XYDataset(Dataset):
    def __init__(
        self,
        root_X,
        root_Y,
        paired=False,
        augment=False,
        transform=None,
        transform_aug=None,
    ):
        self.root_X = root_X
        self.root_Y = root_Y
        self.paired = paired
        self.transform = transform

        self.X_images = os.listdir(root_X)
        self.Y_images = os.listdir(root_Y)
        remove_file(self.X_images, "thumbnail.png")
        remove_file(self.X_images, "blank_patches_list.csv")
        remove_file(self.Y_images, "thumbnail.png")
        remove_file(self.Y_images, "blank_patches_list.csv")

        self.augment = augment
        if self.augment:
            ERR_MESSAGE = "transform_aug is not provided while augment is True"
            assert transform_aug is not None, ERR_MESSAGE
            self.transform_aug = transform_aug

        if paired:
            assert len(self.X_images) == len(self.Y_images)
            self.X_images = sorted(self.X_images)
            self.Y_images = sorted(self.Y_images)
            self.length_dataset = len(self.X_images)

        else:
            self.length_dataset = max(len(self.X_images), len(self.Y_images))
            self.X_len = len(self.X_images)
            self.Y_len = len(self.Y_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        if self.paired:
            X_img = self.X_images[index % self.length_dataset]
            Y_img = self.Y_images[index % self.length_dataset]

        else:
            X_img = self.X_images[index % self.X_len]
            random_y_index = random.randint(0, self.Y_len)
            Y_img = self.Y_images[random_y_index % self.Y_len]

        X_path = os.path.join(self.root_X, X_img)
        Y_path = os.path.join(self.root_Y, Y_img)

        X_img = np.array(Image.open(X_path).convert("RGB"))
        Y_img = np.array(Image.open(Y_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=X_img, image0=Y_img)
            X_img_aug = augmentations["image"]
            Y_img_aug = augmentations["image0"]

        if self.augment:
            double_augmentations = self.transform_aug(
                image=X_img, image0=Y_img
            )
            X_img_double_aug = double_augmentations["image"]
            Y_img_double_aug = double_augmentations["image0"]
        if not self.augment:
            return {"X_img": X_img_aug, "Y_img": Y_img_aug}
        else:
            return {
                "X_img": X_img_aug,
                "Y_img": Y_img_aug,
                "X_img_aug": X_img_double_aug,
                "Y_img_aug": Y_img_double_aug,
            }


class XInferenceDataset(Dataset):
    def __init__(
        self, root_X, transform=None, return_anchor=False, thumbnail=None, pad: int = 16,
    ):
        self.root_X = root_X
        self.transform = transform
        self.return_anchor = return_anchor
        self.thumbnail = thumbnail
        self.pad = pad

        self.X_images = os.listdir(root_X)

        remove_file(self.X_images, "thumbnail.png")
        remove_file(self.X_images, "blank_patches_list.csv")

        if self.return_anchor:
            self.__get_boundary()

        self.length_dataset = len(self.X_images)

    def __get_boundary(self):
        self.y_anchor_num = 0
        self.x_anchor_num = 0
        for X_image in self.X_images:
            y_idx, x_idx, _, _ = Path(X_image).stem.split("_")[:4]
            y_idx = int(y_idx)
            x_idx = int(x_idx)
            self.y_anchor_num = max(self.y_anchor_num, y_idx)
            self.x_anchor_num = max(self.x_anchor_num, x_idx)

    def get_boundary(self):
        assert self.return_anchor
        return (self.y_anchor_num, self.x_anchor_num)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        X_img_name = self.X_images[index]

        X_path = os.path.join(self.root_X, X_img_name)

        X_img = np.array(Image.open(X_path).convert("RGB"))
        X_img = np.pad(
            X_img,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            'reflect',
        )  # pad

        if self.transform:
            augmentations = self.transform(image=X_img)
            X_img = augmentations["image"]

        if self.return_anchor:
            y_idx, x_idx, y_anchor, x_anchor = Path(
                X_img_name
            ).stem.split("_")[:4]
            y_idx = int(y_idx)
            x_idx = int(x_idx)
            return {
                "X_img": X_img,
                "X_path": X_path,
                "y_idx": y_idx,
                "x_idx": x_idx,
                "y_anchor": y_anchor,
                "x_anchor": x_anchor,
            }

        else:
            return {"X_img": X_img, "X_path": X_path}

    def get_thumbnail(self):
        thumbnail_img = np.array(Image.open(self.thumbnail).convert("RGB"))
        if self.transform:
            augmentations = self.transform(image=thumbnail_img)
            thumbnail_img = augmentations["image"]
        return thumbnail_img.unsqueeze(0)


class XPrefetchInferenceDataset(Dataset):
    def __init__(
        self,
        root_X,
        transform=None,
        return_anchor=False,
        thumbnail=None,
        pad: int = 16,
    ):
        self.root_X = root_X
        self.transform = transform
        self.return_anchor = return_anchor
        self.thumbnail = thumbnail
        self.pad = pad

        def custom_sort_key(filename):
            parts = filename.split('_')
            # Extracting the relevant parts of the filename
            first_part = int(parts[1])  # The part after the first underscore
            second_part = int(parts[0])  # The part before the first underscore
            return first_part, second_part

        filenames = os.listdir(root_X)
        remove_file(filenames, "thumbnail.png")
        remove_file(filenames, "blank_patches_list.csv")

        self.X_images = sorted(filenames, key=custom_sort_key)

        if self.return_anchor:
            self.__get_boundary()

        self.length_dataset = len(self.X_images) + self.y_anchor_num + 1 + 2

    def __get_boundary(self):
        self.y_anchor_num = 0
        self.x_anchor_num = 0
        for X_image in self.X_images:
            y_idx, x_idx, _, _ = Path(X_image).stem.split("_")[:4]
            y_idx = int(y_idx)
            x_idx = int(x_idx)
            self.y_anchor_num = max(self.y_anchor_num, y_idx)
            self.x_anchor_num = max(self.x_anchor_num, x_idx)

    def get_boundary(self):
        assert self.return_anchor
        return (self.y_anchor_num, self.x_anchor_num)

    def __len__(self):
        return self.length_dataset

    def get_image(self, index):
        if index < 0 or index >= len(self.X_images):
            X_img = np.zeros((512, 512, 3), dtype=np.uint8)
            X_img = np.pad(
                X_img,
                ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
                'reflect',
            )  # pad
            if self.transform:
                augmentations = self.transform(image=X_img)
                X_img = augmentations["image"]

            return {
                'img': X_img,
                'path': '',
                "y_idx": -1,
                "x_idx": -1,
                "y_anchor": -1,
                "x_anchor": -1,
            }

        X_img_name = self.X_images[index]

        X_path = os.path.join(self.root_X, X_img_name)

        X_img = np.array(Image.open(X_path).convert("RGB"))
        X_img = np.pad(
            X_img,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            'reflect',
        )  # pad
        if self.transform:
            augmentations = self.transform(image=X_img)
            X_img = augmentations["image"]

        y_idx, x_idx, y_anchor, x_anchor = Path(
            X_img_name
        ).stem.split("_")[:4]
        y_idx = int(y_idx)
        x_idx = int(x_idx)

        return {
            'img': X_img,
            "path": X_path,
            "y_idx": y_idx,
            "x_idx": x_idx,
            "y_anchor": y_anchor,
            "x_anchor": x_anchor,
        }

    def __getitem__(self, index):  # start from (0, 0)

        index -= (self.y_anchor_num + 1 + 2)
        index_second = index + self.y_anchor_num + 1 + 2
        indices = [index, index_second]

        images_info = [self.get_image(idx) for idx in indices]
        n = len(images_info)

        return {
            "X_img": images_info[0]['img'],
            "X_path": images_info[0]['path'],
            "y_idx": images_info[0]['y_idx'],
            "x_idx": images_info[0]['x_idx'],
            "y_anchor": images_info[0]['y_anchor'],
            "x_anchor": images_info[0]['x_anchor'],
            "pre_img": [images_info[i]['img'] for i in range(1, n)],
            "pre_path": [images_info[i]['path'] for i in range(1, n)],
            "pre_y_idx": [images_info[i]['y_idx'] for i in range(1, n)],
            "pre_x_idx": [images_info[i]['x_idx'] for i in range(1, n)],
            "pre_y_anchor": [images_info[i]['y_anchor'] for i in range(1, n)],
            "pre_x_anchor": [images_info[i]['x_anchor'] for i in range(1, n)],
        }

    def get_thumbnail(self):
        thumbnail_img = np.array(Image.open(self.thumbnail).convert("RGB"))
        if self.transform:
            augmentations = self.transform(image=thumbnail_img)
            thumbnail_img = augmentations["image"]
        return thumbnail_img.unsqueeze(0)


def get_dataset(config):
    dataset = XYDataset(
        root_X=config["TRAINING_SETTING"]["TRAIN_DIR_X"],
        root_Y=config["TRAINING_SETTING"]["TRAIN_DIR_Y"],
        paired=config["TRAINING_SETTING"]["PAIRED_TRAINING"],
        transform=get_transforms(
            random_crop=config["TRAINING_SETTING"]["RANDOM_CROP_AUG"],
            augment=False,
        ),
    )

    return dataset
