from network import Generator, Discriminator
import torch

G = Generator()

device = "cuda" if torch.cuda.is_available() else "cpu"

noise = torch.randn(64, 100, 1, 1, device="cpu")

# for i in range(1, 4):
#     print(i)
#     out = G(noise)
#     print(out.size())
#     G.grow(depth=i)

out = G(noise)
G.grow(depth=2)

out = G(noise)
G.grow(depth=3)

out = G(noise)
G.grow(depth=4)

out = G(noise)
# G.grow(depth=5)

# out = G(noise)


# for name, layer in G.named_children():
#     print(name, layer)
#     print(type(layer))

# D = Discriminator()

# print(D(out))

# for name, layer in D.named_children():
#     for n, l in layer.named_children():
#         print(n, l)


