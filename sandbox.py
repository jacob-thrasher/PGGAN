from network import Generator, Discriminator
import torch
from torch import nn
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"

G = Generator().to(device)
D = Discriminator(base_size=256).to(device)

img = torch.randn(64, 3, 256, 256, device="cpu").to(device)

out = D(img)
print(out.size())

D.grow(depth=1)
D = D.to(device)
out = D(img)
print(out.size())


