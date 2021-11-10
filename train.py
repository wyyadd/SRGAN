import torch
import srgan
from torch.utils.data import DataLoader
import dataset

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

# generator and discriminator and feature_extractor
generator = srgan.Generator()
discriminator = srgan.Discriminator()
feature_extractor = srgan.FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCELoss()
criterion_content = torch.nn.MSELoss()

if cuda:
    generator.cuda()
    discriminator.cuda()
    feature_extractor.cuda()
    criterion_GAN.cuda()
    criterion_content.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
# Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# data
batch_size = 16
train_dataset = dataset.TrainImageDataset("../dataset/VOC2012", crop_size=96, upscale_factor=4)
train_dataloader = DataLoader(train_dataset, batch_size, num_workers=8, shuffle=True)

# Adversarial ground truths
ones = torch.ones(batch_size).to(device)
zeros = torch.zeros(batch_size).to(device)


def train_loop():
    print("-----start train----")
    for batch, (img_lr, img_hr) in enumerate(train_dataloader):
        img_lr, img_hr = img_lr.to(device), img_hr.to(device)
        # ---------------------
        # ---train generator---
        # ---------------------
        # gen sr
        img_sr = generator(img_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(img_sr), ones)

        # Content loss
        gen_features = feature_extractor(img_sr)
        real_features = feature_extractor(img_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        # -------------------------
        # ---train discriminator
        # -------------------------
        loss_D = criterion_GAN(discriminator(img_hr), ones) + criterion_GAN(discriminator(img_sr.detach()), zeros)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()


if __name__ == '__main__':
    epoch = int(10)
    for i in range(1, epoch + 1):
        train_loop()
