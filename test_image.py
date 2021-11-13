import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, CenterCrop

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

generator = torch.load('../param/srGan_generator_epoch10.pth')
image = Image.open("/home/wyyadd/123.jpg")
image = CenterCrop(32)(image)
image = ToTensor()(image).unsqueeze(0).to(device)
image_sr = generator(image).squeeze(0)
image_sr = ToPILImage()(image_sr)
image_sr.save("test.jpg")
