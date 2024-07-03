import random

import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from scipy import signal
from yaml.loader import SafeLoader


def gkern(kernlen=1, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def get_transforms(random_crop=False, augment=False):
    transforms_list = []

    if random_crop:
        transforms_list.append(
            A.Resize(width=572, height=572, interpolation=cv2.INTER_CUBIC)
        )
        transforms_list.append(A.RandomCrop(width=512, height=512))
    else:
        transforms_list.append(A.Resize(width=512, height=512))

    transforms_list.append(A.HorizontalFlip(p=0.5))

    if augment:
        transforms_list.append(
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            )
        )

    transforms_list.append(
        A.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255
        )
    )
    transforms_list.append(ToTensorV2())

    transforms = A.Compose(
        transforms_list,
        additional_targets={"image0": "image"},
    )
    return transforms


test_transforms = A.Compose(
    [
        # A.Resize(width=512, height=512),  # due to margin padding
        A.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255
        ),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)


def read_yaml_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config


def reverse_image_normalize(img, mean=0.5, std=0.5):
    return img * std + mean


def reduce_size(original_size, patch_size):
    if original_size % patch_size == 0:
        return original_size
    else:
        return (original_size // patch_size) * patch_size


def extend_size(original_size, patch_size):
    if original_size % patch_size == 0:
        return original_size
    else:
        return (original_size // patch_size + 1) * patch_size


def get_kernel(padding=1, gaussian_std=3, mode="constant"):
    kernel_size = padding * 2 + 1
    if mode == "constant":
        kernel = torch.ones(kernel_size, kernel_size)
        kernel = kernel / (kernel_size * kernel_size)

    elif mode == "gaussian":
        kernel = gkern(kernel_size, std=gaussian_std)
        kernel = kernel / kernel.sum()
        kernel = torch.from_numpy(kernel.astype(np.float32))

    else:
        raise NotImplementedError

    return kernel


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (
            max_size > 0
        ), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class ImagePool:
    """This class implements an image buffer that stores previously
    generated images. This buffer enables us to update discriminators
    using a history of generated images rather than the ones produced
    by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer,
            if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously
        stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            # if the buffer is not full;
            # keep inserting current images to the buffer
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                # by 50% chance, the buffer will return a previously
                # stored image, and insert the current image into the buffer
                if p > 0.5:
                    random_id = random.randint(
                        0, self.pool_size - 1
                    )  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)

                # by another 50% chance, the buffer will return the
                # current image
                else:
                    return_images.append(image)

        # collect all the images and return
        return_images = torch.cat(return_images, 0)
        return return_images


def is_blank_patch(image, s_thershold=10, thershold=0.08):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    foreground_proportion = (
        hsv_image[..., 1] > s_thershold
    ).sum() / hsv_image[..., 1].size
    return foreground_proportion < thershold
