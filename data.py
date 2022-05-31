import torch
from torchvision.models import inception_v3
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from network import Identity
from network import IMG_SIZE
from scipy.linalg import sqrtm

inception = inception_v3(pretrained=True)
inception.fc = Identity()
inception.dropout = Identity()
inception.eval()
preprocess = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize((.5, .5, .5), (.5, .5, .5))
])
fid_resize = T.Compose([T.Resize(299)])

class GAN_Dataset(Dataset):
    def __init__(self, d_size, path):
        f, l = process_images(path, d_size)
        self.features = f
        self.length = l

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]

def process_images(path, d_size):
    processed = []
    i = 0
    # to_tensor = T.ToTensor()

    for filename in os.listdir(path):
        if i >= 50000:
            break

        if filename.endswith('jpg'):
            if (i % 1000 == 0):
                print("[PREPROCESSING] Processed {} images".format(i))

        loc = os.path.join(path, filename)
        img = Image.open(loc)
        img = preprocess(img)
        
        processed.append(img)
        
        i+=1

    return processed, len(processed)

def compute_embeddings(real_images, fake_images):
    real_images = fid_resize(real_images)
    fake_images = fid_resize(fake_images)

    real = inception(real_images)
    fake = inception(fake_images)

    return real, fake

#TODO: Fix this
def compute_fid(real_embeddings, fake_embeddings):
    #Compute means and find Euclidean distance
    mu_real = torch.mean(real_embeddings, 1)
    mu_fake = torch.mean(fake_embeddings, 1)
    sq_norm = torch.sum((mu_real - mu_fake) ** 2)

    #Compute covariance matrices
    C_r = torch.cov(real_embeddings)   
    C_f = torch.cov(fake_embeddings)
    C_mean = torch.sqrt(torch.mm(C_r, C_f))
    
    if torch.is_complex(C_mean):
        C_mean = torch.real(C_mean)

    trace = torch.trace(C_r + C_f - 2*C_mean)

    return (sq_norm + trace).item()

#TODO: The real emeddings are computer on the same image set
#each time. This can be optimized by computing them before training
def compute_fid_numpy(real_images, fake_images):
    real_images = fid_resize(real_images)
    fake_images = fid_resize(fake_images)

    real_embeddings = inception(real_images).detach().numpy()
    fake_embeddings = inception(fake_images).detach().numpy()

    mu_real = real_embeddings.mean(axis=0)
    mu_fake = fake_embeddings.mean(axis=0)
    sq_norm = np.sum((mu_real - mu_fake) ** 2)

    C_r = np.cov(real_embeddings, rowvar=False)
    C_f = np.cov(fake_embeddings, rowvar=False)
    C_mean = sqrtm(C_r.dot(C_f))

    if np.iscomplexobj(C_mean):
        C_mean = C_mean.real

    trace = np.trace(C_r + C_f - 2*C_mean)

    return sq_norm + trace





