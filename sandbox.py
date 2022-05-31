from network import Generator, Discriminator
import torch

G = Generator()

device = "cuda" if torch.cuda.is_available() else "cpu"

noise = torch.randn(128, 100, 1, 1, device="cpu")

out = G(noise)


# for name, layer in G.named_children():
#     print(name, layer)
#     print(type(layer))

D = Discriminator()

print(D(out))

# for name, layer in D.named_children():
#     for n, l in layer.named_children():
#         print(n, l)


