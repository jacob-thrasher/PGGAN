from scooby import in_ipython
from sklearn.preprocessing import scale
import torch
from torch import batch_norm, nn
from torchvision.models import inception_v3
from collections import OrderedDict

###################
# HYPERPARAMETERS #
###################
BATCH_SIZE  = 32
LEAKY_SLOPE = 0.2
IMG_SIZE    = 256
SCALE       = 16
LATENT      = 100 #nz
F_MAPS      = 64 #ngf/d
LR          = 0.0002 #0.0002
BETAS       = (0, 0.99)
SCALE_INIT  = 16
###################

scaled_size = IMG_SIZE // 16

#Helper functions
def get_layer_names(model):
    names = []
    for name, _ in model.named_modules():
        if name not in names:
            names.append(name.split('.')[1])

    return names


def initialize_weights(model):
    '''Initialize weights randomly from a Normal distribution with mean=0, std=0.02'''
    classname = model.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0, 0.02)
    elif(classname.find('BatchNorm')) != -1:
        nn.init.normal_(model.weight.data, 1, 0.02)
        nn.init.constant_(model.bias.data, 0)

#Custom layers
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=0, bias=False, do_bn=True):
        super().__init__()
        self.do_bn = do_bn

        self.convt = nn.ConvTranspose2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.convt(x)
        if self.do_bn:
            x = self.bn(x)
        return self.relu(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=0, leaky_slope=0.2, bias=False, do_bn=False):
        super().__init__()
        self.do_bn = do_bn

        self.conv = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(leaky_slope, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.do_bn:
            x = self.bn(x)

        return self.lrelu(x)

class MinibatchStddev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #Compute stddev of all feature maps
        fmap_mean = torch.mean(x)
        sq = torch.square(torch.mean(x - fmap_mean))
        stddev = torch.sqrt(torch.mean(sq))

        #Calculate mean of stddev maps
        avg_stddev = torch.mean(stddev)
        shape = x.size()
        minibatch = torch.tile(avg_stddev, dims=[shape[0], 1, shape[2], shape[3]])

        #cat to input
        return torch.cat([x, minibatch], dim=1)

class Generator(nn.Module):
    def __init__(self, 
                latent_dim=100, 
                f_maps=64, 
                scale_init=16,
                d_size=256):
                
        super(Generator, self).__init__()

        self.f_maps = f_maps
        self.scale_init = scale_init
        self.alpha = 0
        self.depth = 0

        self.rgb = self.to_rgb(depth=1)

        self.upsample = nn.Upsample(size=(d_size, d_size), mode='nearest')
        self.base = nn.Sequential()
        self.base.add_module('conv_0', ConvTranspose(latent_dim, f_maps*scale_init, kernel_size=2, stride=1))

        self.old_head = nn.Sequential(OrderedDict([
            ('to_rgb', self.to_rgb(depth=0)),
            ('upsample', nn.Upsample(scale_factor=2, mode='nearest'))
        ]))

        s = int(self.scale_init / 2)
        self.new_head = nn.Sequential(OrderedDict([
            ('conv', ConvTranspose(self.f_maps*self.scale_init, self.f_maps*s, kernel_size=4, stride=2, padding=1)),
            ('to_rgb', self.to_rgb(depth=1))
        ]))

    def to_rgb(self, depth):
        s = int(self.scale_init / (2 ** depth))
        layers = []
        layers.append(nn.ConvTranspose2d(self.f_maps*s, 3, kernel_size=5, stride=1, padding=2))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def grow(self, depth):
        depth += 1
        #Append new_head.conv to base
        module = self.new_head.conv
        self.base.add_module(f'conv_{depth}', module)
        self.base[-1].load_state_dict(module.state_dict())

        #Replace old head:
        module = self.new_head.to_rgb
        self.old_head = nn.Sequential(OrderedDict([
            ('to_rgb', module),
            ('upsample', nn.Upsample(scale_factor=2, mode='nearest'))
        ]))
        self.old_head[0].load_state_dict(module.state_dict())

        #Replace new head:
        s = int(self.scale_init / (2 ** depth))
        self.new_head = nn.Sequential(OrderedDict([
            ('conv', ConvTranspose(self.f_maps*(s*2), self.f_maps*s, kernel_size=4, stride=2, padding=1)),
            ('to_rgb', self.to_rgb(depth))
        ]))

    def forward(self, x):
        x = self.base(x)
        y = self.old_head(x)
        z = self.new_head(x)

        return self.upsample(((1 - self.alpha) * y) + (self.alpha * z))
        # return ((1 - self.alpha) * y) + (self.alpha * z)

class Discriminator(nn.Module):
    def __init__(self, 
                f_maps=64,
                base_size=128,
                channels = 3):
        super(Discriminator, self).__init__()

        self.f_maps = f_maps
        self.downsample_factor = base_size // 4 # 4 = initial img size
        self.alpha = 0
        self.depth = 0

        self.downsample = nn.AvgPool2d(kernel_size=self.downsample_factor)
        self.base = nn.Sequential()
        self.base.add_module('from_rgb', Conv(channels, self.f_maps, kernel_size=4, stride=2, padding=1, do_bn=True))
        self.minibatch = MinibatchStddev()

        self.new_head = nn.Sequential(OrderedDict([
            ('conv', Conv(self.f_maps, self.f_maps*2, kernel_size=4, stride=2, padding=1, do_bn=True)),
            ('out', self.to_output(depth=1))
        ]))
        self.old_head = nn.Sequential(OrderedDict([
            ('pool', nn.AvgPool2d(kernel_size=2)),
            ('out', self.to_output(depth=0))
        ]))

    def to_output(self, depth):
        s = 2 ** depth
        layers = [
            # MinibatchStddev(),
            nn.Flatten(),
            nn.Linear(self.f_maps*s, 1),
            nn.Sigmoid()
        ]
        return nn.Sequential(*layers)

    def grow(self, depth):
        # depth += 1
        # append new_head.conv to base
        module = self.new_head.conv
        self.base.add_module(f'conv_{depth}', module)
        self.base[-1].load_state_dict(module.state_dict())

        #Replace old head:
        module = self.new_head.out
        self.old_head = nn.Sequential(OrderedDict([
            ('pool', nn.AvgPool2d(kernel_size=2)),
            ('out', module)
        ]))
        self.old_head[-1].load_state_dict(module.state_dict())

        #Replace new head:
        s = 2 ** depth
        self.new_head = nn.Sequential(OrderedDict([
            ('conv', Conv(self.f_maps*s, self.f_maps*(2*s), kernel_size=4, stride=2, padding=1, do_bn=True)),
            ('out', self.to_output(depth=depth+1))
        ]))

        self.downsample = nn.AvgPool2d(kernel_size=(self.downsample_factor // (2 ** depth)))

    def forward(self, x):
        x = self.downsample(x)
        x = self.base(x)
        y = self.old_head(x)
        z = self.new_head(x)

        return ((1 - self.alpha) * y) + (self.alpha * z)



