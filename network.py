from scooby import in_ipython
import torch
from torch import batch_norm, nn
from torchvision.models import inception_v3
from collections import OrderedDict

###################
# HYPERPARAMETERS #
###################
BATCH_SIZE  = 128
LEAKY_SLOPE = 0.2
IMG_SIZE    = 64
SCALE       = 16
LATENT      = 100 #nz
F_MAPS      = 64 #ngf/d
LR          = 0.0002 #0.0002
BETAS       = (0.5, 0.999)
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

def conv_transpose(in_channels, out_channels, k_size=5, stride=2, padding=0, bias=False, bn=True):
    if bn:
        layers = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, stride=stride, 
                            kernel_size=k_size, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    else:
        layers = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, stride=stride, 
                            kernel_size=k_size, padding=padding, bias=bias),
        nn.ReLU(inplace=True)
    )

    return layers

def conv(in_channels, out_channels, k_size=5, stride=2, padding=0, bias=False, bn=True):

    if bn:
        layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=k_size, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(LEAKY_SLOPE, inplace=True)
    )

    else:
        layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=k_size, padding=padding, bias=bias),
        nn.LeakyReLU(LEAKY_SLOPE, inplace=True)
    )

    return layers

#Custom layers
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

#Network
#TODO: Depth=5 fails, this is due to the scalar math I do, fix that
class Generator(nn.Module):
    def __init__(self, 
                latent_dim=100, 
                f_maps=64, 
                scale_init=16, 
                a_rate=0.2):
                
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.f_maps = f_maps
        self.scale_init = scale_init
        self.alpha = 0
        self.a_rate = a_rate
        self.depth = 0

        self.base = nn.Sequential()
        self.base.add_module('conv1', conv_transpose(latent_dim, f_maps*scale_init, k_size=2, stride=1))

        self.old_head = nn.Sequential(OrderedDict([
            ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ('to_rgb', self.to_rgb(depth=0))
        ]))

        # self.new_head = nn.Sequential(OrderedDict([
        #     ('conv', conv_transpose(f_maps*scale_init, f_maps*int(scale_init/2), k_size=4, stride=2, padding=1)),
        #     ('to_rgb', self.to_rgb(depth=1))
        # ]))
        self.new_head = nn.Sequential(OrderedDict([
            ('conv', self.create_new_block(depth=1)),
            ('to_rgb', self.to_rgb(depth=1))
        ]))

    def to_rgb(self, depth):
        s = int(self.scale_init / (2 ** depth))
        layers = []
        layers.append(nn.ConvTranspose2d(self.f_maps*s, 3, kernel_size=5, stride=1, padding=2))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def create_new_block(self, depth):
        s = int(self.scale_init / (2 ** depth))
        name = f'convt_{depth}'
        layers = conv_transpose(self.f_maps*(s*2), self.f_maps*s, k_size=4, stride=2, padding=1)
        return layers
        # return layers, name

    def grow(self, depth):
        #Append new_head.conv to base
        module = self.new_head.conv
        self.base.add_module(f'conv_{depth}', module)
        self.base[-1].load_state_dict(module.state_dict())

        #Replace old head:
        module = self.new_head.to_rgb
        self.old_head = nn.Sequential(OrderedDict([
            ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ('to_rgb', module)
        ]))
        self.old_head[-1].load_state_dict(module.state_dict())

        #Replace new head:
        self.new_head = nn.Sequential(OrderedDict([
            ('conv', self.create_new_block(depth)),
            ('to_rgb', self.to_rgb(depth))
        ]))

    def forward(self, x):
        x = self.base(x)
        y = self.old_head(x)
        z = self.new_head(x)

        print(self.base)
        print(y.size(), z.size())
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.conv_0  = conv(3, F_MAPS, k_size=4, stride=2, padding=1, bn=False)
        # self.conv_1  = conv(F_MAPS, F_MAPS*2, k_size=4, stride=2, padding=1)

        # self.conv_2  = conv(F_MAPS*2, F_MAPS*4, k_size=4, stride=2, padding=1)
        # self.conv_3  = conv(F_MAPS*4, F_MAPS*8, k_size=4, stride=2, padding=1)
        # self.out     = conv(F_MAPS*8, 1, k_size=4, stride=1, padding=0, bn=False)
        # self.flatten = nn.Flatten()
        # self.dense   = nn.Linear(4*4*F_MAPS*8, 1)

        init = conv(3, F_MAPS, k_size=4, stride=2, padding=1, bn=False)
        self.model = nn.Sequential()
        self.model.add_module('from_rgb', init)
        self.model.add_module('out', self.to_output(depth=0))

    def to_output(self, depth):
        s = depth ** 2
        layers = []
        layers.append(conv(F_MAPS*s, F_MAPS*(2*s), k_size=4, stride=2, padding=1, bn=False))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def create_new_block(self, depth):
        s = depth ** 2
        name = f'conv_{depth}'
        return conv(F_MAPS*s, F_MAPS*(2*s), k_size=4, stride=2, padding=1), name

    def grow(self, depth):
        new_model = nn.Sequential()
        old_out = nn.Sequential()

        for _, old_model in self.model.named_children():
            for name, module in old_model.named_children():
                if not name == 'out':        #Grab old base
                    new_model.add_module(name, module)
                    new_model[-1].load_state_dict(module.state_dict())
                else:     
                    old_out = None          #Just in case               
                    old_out = module        #Grab old output

        prev_block = nn.Sequential()
        prev_block.add_module('downsample', nn.AvgPool2d(kernel_size=2))
        prev_block.add_module('old_out', old_out)

        new_layers, lname = self.create_new_block(depth)
        new_block = nn.Sequential()
        new_block.add_module(lname, new_layers)


        return

    def flush_network(self):
        #Same as in G
        return

    def forward(self, x):
        return self.model(x)



