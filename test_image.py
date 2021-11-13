import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, CenterCrop, RandomCrop, Compose, Normalize

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

test_transform_before = Compose([
    RandomCrop(16),
    ToTensor()
])

test_transform_after = Compose([
    Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444]),
    ToPILImage()
])

generator = torch.load('../param/srGan_generator_epoch30.pth')
generator.eval()
image = Image.open("/home/wyyadd/2.png")
image.save("test1.jpg")
image = torch.unsqueeze(test_transform_before(image), 0).to(device)
image_sr = generator(image).squeeze(0)
image_sr = test_transform_after(image_sr)
image_sr.save("test.jpg")
