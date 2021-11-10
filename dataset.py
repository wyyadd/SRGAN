import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, transforms, \
    InterpolationMode


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=InterpolationMode.BICUBIC),
        ToTensor()
    ])


class TrainImageDataset(Dataset):
    def __init__(self, root, crop_size, upscale_factor, train=True):
        if train:
            self.files = glob.glob(root + "/train/*.jpg")
        else:
            self.files = glob.glob(root + "/val/*.jpg")
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        img_hr = self.hr_transform(img)
        img_lr = self.lr_transform(img_hr)
        return img_lr, img_hr

    def __len__(self):
        return len(self.files)

# x = TrainImageDataset("../dataset/VOC2012", crop_size=16, upscale_factor=4)
# train_dataloader = DataLoader(x, batch_size=5, shuffle=True, num_workers=8)
# x, y = next(iter(train_dataloader))
# print(x, y)
