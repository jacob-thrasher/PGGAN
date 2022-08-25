from network import Generator, Discriminator
import torch

G = Generator()
D = Discriminator()

device = "cuda" if torch.cuda.is_available() else "cpu"

# noise = torch.randn(64, 100, 1, 1, device="cpu")
img = torch.randn(64, 3, 4, 4, device=device)
x = D(img)
print(x.size())

D.grow(depth=1)
img = torch.randn(64, 3, 8, 8, device=device)
x = D(img)
print(x.size())

D.grow(depth=2)
img = torch.randn(64, 3, 16, 16, device=device)
x = D(img)
print(x.size())

