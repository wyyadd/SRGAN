import torch
import srgan
from torch.utils.data import DataLoader
from torchvision import datasets

cuda = torch.cuda.is_available()

# generator and discriminator and feature_extractor
generator = srgan.Generator()
discriminator = srgan.Discriminator()
feature_extractor = srgan.FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator.cuda()
    discriminator.cuda()
    feature_extractor.cuda()
    criterion_GAN.cuda()
    criterion_content.cuda()
    
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

train_data = datasets.ImageNet()



