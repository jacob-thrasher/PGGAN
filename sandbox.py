from network import Generator, Discriminator
import torch
from torch import nn
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"

G = Generator().to(device)
D = Discriminator(base_size=256).to(device)

# img = torch.randn(64, 3, 256, 256, device="cpu").to(device)
noise = torch.randn(64, 100, 1, 1).to(device)

G.grow(depth=1)
G.to(device)

out, (x, y, z) = G(noise)

save_image(out, 'out.png')
save_image(y, 'y.png')
save_image(z, 'z.png')

G.alpha = 0.5

out, (x, y, z) = G(noise)

save_image(out, 'out.png')
save_image(y, 'y.png')
save_image(z, 'z.png')


