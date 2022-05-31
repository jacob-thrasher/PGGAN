import torch

def generator_minimize_loss(fake_output):
    return -1 * torch.mean(torch.log(1 - fake_output))

def generator_maximize_loss(fake_output):
    return torch.mean(torch.log(fake_output))

def discriminator_loss(real_output, fake_output):
    real_loss = torch.log(real_output)
    fake_loss = torch.log(1 - fake_output)
    total_loss = real_loss + fake_loss

    return torch.mean(total_loss)