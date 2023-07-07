import random
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
import torch
import numpy as np
from typing import List, Tuple
from torchvision.transforms import functional as TF


class RandomRotate(object):
    """Implementation of random rotation.

    Randomly rotates an input image by a fixed angle. By default, we rotate
    the image by 90 degrees with a probability of 50%.

    This augmentation can be very useful for rotation invariant images such as
    in medical imaging or satellite imaginary.

    Attributes:
        prob:
            Probability with which image is rotated.
        angle:
            Angle by which the image is rotated. We recommend multiples of 90
            to prevent rasterization artifacts. If you pick numbers like
            90, 180, 270 the tensor will be rotated without introducing
            any artifacts.

    """

    def __init__(self, prob: float = 0.5, angle: int = 90):
        self.prob = prob
        self.angle = angle

    def __call__(self, sample):
        """Rotates the images with a given probability.

        Args:
            sample:
                PIL image which will be rotated.

        Returns:
            Rotated image or original image.

        """
        prob = np.random.random_sample()
        if prob < self.prob:
            sample = TF.rotate(sample, self.angle)
        return sample


class Cutout(object):
    """
    Randomly mask out one or more patches from an image.
    https://arxiv.org/abs/1708.04552
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class GaussianBlur(object):
    """Gaussian blur augmentation: https://github.com/facebookresearch/moco/"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization(object):
    """
    ..
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class MNISTTransform:
    """
    mnist transform
    """
    def __init__(self):
        self.mean = (0.1307,)
        self.std = (0.3081,)

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomPerspective(p=0.7),
            # GaussianBlur(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)])

        self.transform_prime = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomPerspective(p=0.3),
            # GaussianBlur(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


class BasicTransform:
    """
        Basic Transform
    """
    def __init__(self, mean: List, std: List, input_shape: int):

        self.mean, self.std = mean, std
        self.transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __call__(self, x):
        return self.transform(x)


class CIFAR10Transform:
    """
        cifar10 transforms
    """
    def __init__(self, mean: List, std: List, input_shape: int = 32, multi_viewed: bool = True):
        self.mean, self.std = mean, std
        self.mv = multi_viewed
        self.transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.RandomCrop(input_shape, padding=4),  # important for baseline supervised ~ CutOut Paper
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            # Cutout(n_holes=1, length=16),
        ])
        self.contrastive_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_shape),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __call__(self, x):
        if self.mv:
            return self.contrastive_transform(x), \
                   self.contrastive_transform(x)
        else:
            return self.transform(x)


class FMNISTTransform:
    """
    Fashion MNIST Transform
    """
    def __init__(self, mean: Tuple, std: Tuple, input_shape: int = 28, multi_viewed: bool = True):
        self.mean = mean
        self.std = std
        self.mv = multi_viewed

        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(input_shape),
            # transforms.RandomHorizontalFlip(p=0.5),
            # # transforms.RandomPerspective(p=0.1),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        self.contrastive_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_shape),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            # transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomPerspective(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # RandomRotate(prob=0.5, angle=90),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __call__(self, x):
        if self.mv:
            return self.contrastive_transform(x), \
                self.contrastive_transform(x)
        else:
            return self.transform(x)


class STLTransform:
    """
    stl transform
    """
    def __init__(self, mean: List, std: List, input_shape: int = 96, multi_viewed: bool = True):
        self.mean, self.std = mean, std
        self.mv = multi_viewed

        self.transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.RandomCrop(input_shape, padding=4),  # important for baseline supervised ~ CutOut Paper
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            Cutout(n_holes=1, length=16),
        ])

        self.contrastive_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=96),
            transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])

    def __call__(self, x):
        if self.mv:
            return self.contrastive_transform(x), self.contrastive_transform(x)
        else:
            return self.transform(x)


class ImageNetTransform:
    """
        ImageNet Transform
    """
    def __init__(self, mean: List, std: List, input_shape: int = 224):

        self.mean, self.std = mean, std

        # self.transform = transforms.Compose([
        #     transforms.RandomResizedCrop(input_shape, interpolation=Image.BICUBIC),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        #     Solarization(p=0.0),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        if self.mv:
            return self.transform(x), self.transform(x)
        else:
            return self.transform(x)
