import glob

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, transforms, \
    InterpolationMode


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform():
    return Compose([
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        Resize(crop_size // upscale_factor, interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TrainImageDataset(Dataset):
    def __init__(self, root, crop_size, upscale_factor, train=True):
        self._upscale_factor = upscale_factor
        if train:
            self.files = glob.glob(root + "/train/*.jpg")
        else:
            self.files = glob.glob(root + "/val/*.jpg")
        self._crop_size = calculate_valid_crop_size(crop_size, self._upscale_factor)
        self.hr_transform = train_hr_transform()
        self.lr_transform = train_lr_transform(self._crop_size, self._upscale_factor)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        img = RandomCrop(self._crop_size)(img)
        img_hr = self.hr_transform(img)
        img_lr = self.lr_transform(img)
        return img_lr, img_hr

    def __len__(self):
        return len(self.files)
