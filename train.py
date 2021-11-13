import torch
import srgan
from torch.utils.data import DataLoader
import dataset
import matplotlib.pyplot as plt

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

epoch = int(31)
# generator and discriminator and feature_extractor
generator = srgan.Generator()
# generator = torch.load('../param/srGan_generator_epoch30.pth')
discriminator = srgan.Discriminator()
# discriminator = torch.load('../param/srGan_discriminator_epoch30.pth')
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
optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-5)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-5)
# Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# data
batch_size = 32
train_dataset = dataset.TrainImageDataset("../dataset/VOC2012", crop_size=96, upscale_factor=4)
train_dataloader = DataLoader(train_dataset, batch_size, num_workers=8, shuffle=True)


def train_loop(train_epoch):
    print("-----start train----")
    generator.train()
    discriminator.train()
    data_len = len(train_dataloader)
    g_loss = []
    d_loss = []
    for batch, (img_lr, img_hr) in enumerate(train_dataloader):
        img_lr, img_hr = img_lr.to(device), img_hr.to(device)
        # Adversarial ground truths
        ones = torch.ones(img_hr.size()[0]).to(device)
        zeros = torch.zeros(img_hr.size()[0]).to(device)
        # ---------------------
        # ---train generator---
        # ---------------------
        # gen sr
        img_sr = generator(img_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(img_sr), ones)

        # Content loss
        gen_features = feature_extractor(img_sr.detach())
        real_features = feature_extractor(img_hr)
        loss_content = criterion_content(img_sr, img_hr) + 0.006 * criterion_content(gen_features, real_features)

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        generator.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        # -------------------------
        # ---train discriminator
        # -------------------------
        loss_D = criterion_GAN(discriminator(img_hr), ones) + criterion_GAN(discriminator(img_sr.detach()), zeros)
        discriminator.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ------------------------
        # ----------log-----------
        # ------------------------
        if batch % 100 == 0:
            g_loss.append(loss_G.item())
            d_loss.append(loss_D.item())
            print("epoch {}, G_loss: {:.6f}, D_loss: {:.6f}, {}/{}".format(train_epoch, loss_G.item(), loss_D.item(),
                                                                           batch,
                                                                           data_len))


def test_loop():
    print("----start evaluate----")
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        pass


if __name__ == '__main__':
    for i in range(epoch, epoch + 20):
        train_loop(i)
        # test_loop()
        if i % 10 == 0:
            torch.save(generator, '../param/srGan_generator_epoch{}.pth'.format(i))
            torch.save(discriminator, '../param/srGan_discriminator_epoch{}.pth'.format(i))
