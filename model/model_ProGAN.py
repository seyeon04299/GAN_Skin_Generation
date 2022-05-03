import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
from torchvision import models
import sys
import model.norm as module_norm
from utils import num_flat_features
from numpy import prod

## Factors for discriminator and generator channel changes
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8
    
    def forward(self,x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
    
    
class WSConv2d(nn.Module):          #Weighted Scaled convolutional layers
    '''
    Weighted scaled Conv2d (Equalized Learning Rate - Section 4.1 of Paper)
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,gain=2):
        super(WSConv2d,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain/(in_channels*(kernel_size**2)))**0.5

        # self.bias = self.conv.bias
        # self.conv.bias = None

        # initialize conv layer
        self.conv.weight.data.normal_(0,1)
        self.conv.bias.data.fill_(0)
        # nn.init.normal_(self.conv.weight)
        # nn.init.zeros_(self.bias)


    def forward(self,x):
        out = self.conv(x)*self.scale
        return(out)

class WSLinear(nn.Module):
    '''
    Weighted Scale Linear (Equalized Learning Rate)
    '''
    def __init__(self,in_channels,out_channels,gain=2):
        super(WSLinear,self).__init__()
        self.linear = nn.Linear(in_channels,out_channels,bias=True)
        self.scale = (gain/(prod(self.linear.weight.size()[1:])))**0.5
        
        # Initialize weights
        self.linear.weight.data.normal_(0,1)
        self.linear.bias.data.fill_(0)

    def forward(self,x):
        out = self.linear(x)*self.scale
        return(out)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

        self.selu = nn.SELU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.use_pn:
            x = self.leaky(self.conv1(x))
            x = self.pn(x)
            x = self.leaky(self.conv2(x))
            x = self.pn(x)
        
        else:
            x = self.conv1(x)
            x = self.pn(x)
            x = self.selu(x)
            x = self.conv2(x)
            x = self.pn(x)
            x = self.selu(x)
        return x


class Generator_ProGAN(nn.Module):
    def __init__(self, nz, ngf, img_channels=3, factors =[]):
        super(Generator_ProGAN, self).__init__()
        
        # initial takes 1x1 -> 4x4
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(nz, ngf, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        self.initial_rgb = WSConv2d(
            ngf, img_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]),
        )

        for i in range(
            len(factors) - 1
        ):  # -1 to prevent index error because of factors[i+1]
            conv_in_c = int(ngf * factors[i])
            conv_out_c = int(ngf * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=True))
            self.rgb_layers.append(
                WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, upscaled, generated):
        # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):     # steps=0 (4x4), steps=1 (8x8), ...
        out = self.initial(x)   # 4x4

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        # The number of channels in upscale will stay the same, while
        # out which has moved through prog_blocks might change. To ensure
        # we can convert both to rgb we use different rgb_layers
        # (steps-1) and steps for upscaled, out respectively

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)



class Discriminator_ProGAN(nn.Module):
    def __init__(self, ndf, img_channels=3, factors =[]):
        super(Discriminator_ProGAN,self).__init__()

        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)
        # Work Backwords from factor
        for i in range(len(factors)-1, 0,-1):
            conv_in = int(ndf*factors[i])
            conv_out = int(ndf*factors[i-1])
            self.prog_blocks.append(ConvBlock(conv_in,conv_out,use_pixelnorm=True))
            self.rgb_layers.append(WSConv2d(img_channels,conv_in,kernel_size=1,stride=1,padding=0))
        
        
        # RGB layer for 4x4 input size, to "mirror" the generator initial_rgb
        self.initial_rgb = WSConv2d(
            img_channels, ndf, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.initial_rgb)

        # DownSampling
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block for 4x4 input size
        self.final_block_conv = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            WSConv2d(ndf + 1, ndf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(ndf, ndf, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(
                ndf, 1, kernel_size=1, padding=0, stride=1
            ),
        )

        self.final_block = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            WSConv2d(ndf + 1, ndf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(ndf, ndf, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2)
        )

        self.decision_layer = nn.Sequential(
            WSLinear(ndf,1)
        )



    def fade_in(self, alpha, downscaled, out):
        """Used to fade in downscaled using avg pooling and output from CNN"""
        # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        # we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        # will get information about the variation in the batch/image
        return torch.cat([x, batch_statistics], dim=1)


    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks)-steps
        # Convert from rgb as initial step
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps==0:
            out = self.minibatch_std(out)
            # out = self.final_block_conv(out).view(out.shape[0],-1)
            out = self.final_block(out)
            out = out.view(-1,num_flat_features(out))
            out = self.decision_layer(out)
            return out

        downscaled = self.leaky(self.rgb_layers[cur_step+1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))

        out = self.fade_in(alpha, downscaled, out)
    
        for step in range(cur_step+1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
        
        out = self.minibatch_std(out)
        # out = self.final_block_conv(out).view(out.shape[0], -1)
        out = self.final_block(out)
        out = out.view(-1,num_flat_features(out))
        out = self.decision_layer(out)

        return out
