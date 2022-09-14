from network import Generator, Discriminator
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

G = Generator().to(device)
D = Discriminator(base_size=256).to(device)

# noise = torch.randn(64, 100, 1, 1, device="cpu")
img = torch.randn(64, 3, 256, 256, device=device)
x = D(img)
print(x.size())

D.grow(depth=1)
D.to(device)
x = D(img)
print(x.size())

D.grow(depth=2)
D.to(device)
x = D(img)
print(x.size())



