import torch
from torch import batch_norm, nn
from torchvision.models import inception_v3

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
SCALE_INIT  = 8
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

#Network
#TODO: Reduce initial output from 8x8 to 4x4
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        init = conv_transpose(LATENT, F_MAPS*SCALE_INIT, k_size=4, stride=1)
        self.model = nn.Sequential()
        self.model.add_module('base', init)
        self.model.add_module('to_rgb', self.to_rgb(depth=0))

    def to_rgb(self, depth):
        s = int(SCALE_INIT / (2 ** depth))
        layers = []
        layers.append(nn.ConvTranspose2d(F_MAPS*s, 3, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def create_new_block(self, depth):
        s = int(SCALE_INIT / (2 ** depth))
        name = f'convt_{depth}'
        layers = conv_transpose(F_MAPS*(s*2), F_MAPS*s, k_size=4, stride=2, padding=1)
        return layers, name

    def grow(self, depth):
        new_model = nn.Sequential()
        old_rgb = nn.Sequential()

        #The actual layers of the model are wrapped into a seq 
        #network itself, so we need to access named_children twice :(
        for _, old_model in self.model.named_children():
            for name, module in old_model.named_children():
                if not name == 'to_rgb':        #Grab old base
                    new_model.add_module(name, module)
                    new_model[-1].load_state_dict(module.state_dict())
                else:     
                    old_rgb.add_module('to_rgb', module)     
                    old_rgb[-1].load_state_dict(module.state_dict())
        
        # Shove upsampling layer btwn base and output
        prev_block = nn.Sequential()
        prev_block.add_module('upsample', nn.Upsample(scale_factor=2, mode='nearest'))
        prev_block.add_module('old_rgb', old_rgb)

        # Create new convolutional block
        new_layers, lname = self.create_new_block(depth)
        new_block = nn.Sequential()
        new_block.add_module(lname, new_layers)
        new_block.add_module('to_rgb', self.to_rgb(depth))

        #Compute weighted sum
        new_model.add_module('weighted_sum', WeightedSum(prev_block, new_block))
        self.model = None
        self.model = new_model
        return

    def flush_network(self):
        #Here we need to remove the WeightedSum layer and replace
        #It with only the new output

        #Call this when alpha==1
        return

    def forward(self, x):
        return self.model(x)

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

#Custom layers
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class WeightedSum(nn.Module):
    def __init__(self, old_block, new_block):
        self.old = old_block
        self.new = new_block
        self.alpha = 0

    def update_alpha(self, delta):
        self.alpha += delta
        self.alpha = min(1, self.alpha)

    def forward(self, x):
        old_out = self.old(x)
        new_out = self.new(x)

        return old_out*(1-self.alpha) + new_out*self.alpha

