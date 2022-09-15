from network import Generator, Discriminator
import torch
from torch import nn
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"

G = Generator().to(device)
D = Discriminator(base_size=256).to(device)

noise = torch.randn(64, 100, 1, 1, device="cpu").to(device)
# G.grow(depth=2)
# G = G.to(device)

out = G(noise)
save_image(out, 'out1.png')


G.alpha = 0.25

out = G(noise)
save_image(out, 'out2.png')

G.alpha = .75

out = G(noise)
save_image(out, 'out3.png')




