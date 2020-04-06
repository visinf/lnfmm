import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_

''' Code based on SaGAN.py from https://github.com/voletiv/self-attention-GAN-pytorch.git'''
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))


def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out


class ConditionalBatchNorm2d(nn.Module):
    # https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    def __init__(self, num_features, t2i_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)
        self.embed = nn.Linear(t2i_dim, num_features * 2)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        #self.embed.weight.data[:, :num_features].fill_(1.)  # Initialize scale to 1
        #self.embed.weight.data[:, num_features:].zero_()  # Initialize bias at 0

    def forward(self, x, t2i):
        out = self.bn(x)
        gamma, beta = self.embed(t2i).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t2i_dim):
        super(GenBlock, self).__init__()
        self.cond_bn1 = ConditionalBatchNorm2d(in_channels, t2i_dim)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.cond_bn2 = ConditionalBatchNorm2d(out_channels, t2i_dim)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t2i):
        x0 = x

        x = self.cond_bn1(x, t2i)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.snconv2d1(x)
        x = self.cond_bn2(x, t2i)
        x = self.relu(x)
        x = self.snconv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class Generator(nn.Module):
    """Generator."""

    def __init__(self, z_dim, g_conv_dim, t2i_dim):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.g_conv_dim = g_conv_dim
        self.snlinear0 = snlinear(in_features=(t2i_dim + z_dim), out_features=g_conv_dim*32*4*4)
        self.block1 = GenBlock(g_conv_dim*32, g_conv_dim*32, t2i_dim)
        self.block2 = GenBlock(g_conv_dim*32, g_conv_dim*16, t2i_dim)
        self.block3 = GenBlock(g_conv_dim*16, g_conv_dim*8, t2i_dim)
        self.block4 = GenBlock(g_conv_dim*8, g_conv_dim*4, t2i_dim)
        self.block5 = GenBlock(g_conv_dim*4, g_conv_dim*2, t2i_dim)
        self.self_attn = Self_Attn(g_conv_dim*4)
        self.block6 = GenBlock(g_conv_dim*2, g_conv_dim, t2i_dim)
        self.bn = nn.BatchNorm2d(g_conv_dim, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=g_conv_dim, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        self.apply(init_weights)

    def forward(self, z, t2i):
        # n x z_dim
        act0 = self.snlinear0(torch.cat([z,t2i], dim=1))            # n x g_conv_dim*16*4*4
        act0 = act0.view(-1, self.g_conv_dim*32, 4, 4) # n x g_conv_dim*16 x 4 x 4
        act1 = self.block1(act0, t2i)    # n x g_conv_dim*16 x 8 x 8
        act2 = self.block2(act1, t2i)    # n x g_conv_dim*8 x 16 x 16
        act3 = self.block3(act2, t2i)    # n x g_conv_dim*4 x 32 x 32         # n x g_conv_dim*4 x 32 x 32
        act4 = self.block4(act3, t2i)    # n x g_conv_dim*2 x 64 x 64
        act4 = self.self_attn(act4)
        act5 = self.block5(act4, t2i)    # n x g_conv_dim  x 128 x 128
        act6 = self.block6(act5, t2i)
        act6 = self.bn(act6)                # n x g_conv_dim  x 128 x 128
        act6 = self.relu(act6)              # n x g_conv_dim  x 128 x 128
        act7 = self.snconv2d1(act6)         # n x 3 x 128 x 128
        act7 = self.tanh(act7)              # n x 3 x 128 x 128
        return act7

class Generator64(nn.Module):
    """Generator."""

    def __init__(self, z_dim, g_conv_dim, t2i_dim):
        super(Generator64, self).__init__()

        self.z_dim = z_dim
        self.g_conv_dim = g_conv_dim
        self.snlinear0 = snlinear(in_features=(z_dim), out_features=g_conv_dim*16*4*4)#t2i_dim + 
        self.block1 = GenBlock(g_conv_dim*16, g_conv_dim*8, t2i_dim)
        self.block2 = GenBlock(g_conv_dim*8, g_conv_dim*4, t2i_dim)
        self.self_attn = Self_Attn(g_conv_dim*4)
        self.block3 = GenBlock(g_conv_dim*4, g_conv_dim*2, t2i_dim)
        self.block4 = GenBlock(g_conv_dim*2, g_conv_dim, t2i_dim)
        self.bn = nn.BatchNorm2d(g_conv_dim, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=g_conv_dim, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        self.apply(init_weights)

    def forward(self, z, t2i):
        # n x z_dim
        act0 = self.snlinear0(z)            # n x g_conv_dim*16*4*4. # torch.cat([z,t2i], dim=1)
        act0 = act0.view(-1, self.g_conv_dim*16, 4, 4) # n x g_conv_dim*16 x 4 x 4
        act1 = self.block1(act0, t2i)    # n x g_conv_dim*16 x 8 x 8
        act2 = self.block2(act1, t2i)    # n x g_conv_dim*8 x 16 x 16
        act3 = self.self_attn(act2)         # n x g_conv_dim*4 x 32 x 32
        act4 = self.block3(act3, t2i)    # n x g_conv_dim*2 x 64 x 64
        act5 = self.block4(act4, t2i)
        act5 = self.bn(act5)                # n x g_conv_dim  x 128 x 128
        act5 = self.relu(act5)              # n x g_conv_dim  x 128 x 128
        act6 = self.snconv2d1(act5)         # n x 3 x 128 x 128
        act6 = self.tanh(act6)              # n x 3 x 128 x 128
        return act6       




class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, d_conv_dim, t2i_dim):
        super(Discriminator, self).__init__()
        self.d_conv_dim = d_conv_dim
        self.opt_block1 = DiscOptBlock(3, d_conv_dim)
        self.block1 = DiscBlock(d_conv_dim, d_conv_dim*2)
        self.self_attn = Self_Attn(d_conv_dim*2)
        self.block2 = DiscBlock(d_conv_dim*2, d_conv_dim*4)
        self.block3 = DiscBlock(d_conv_dim*4, d_conv_dim*8)
        self.block4 = DiscBlock(d_conv_dim*8, d_conv_dim*16)
        self.block5 = DiscBlock(d_conv_dim*16, d_conv_dim*16)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=d_conv_dim*16, out_features=1)
        #self.snlinear2 = snlinear(in_features=d_conv_dim*16, out_features=1)
        self.snlinear2 = snlinear(in_features=t2i_dim, out_features=d_conv_dim*16)
        #self.sn_embedding1 = sn_embedding(num_classes, d_conv_dim*16)

        self.cond_discriminator_mlp = CondDiscriminatorMLP(txt_dim=t2i_dim, im_dim=d_conv_dim*16, out_dim=d_conv_dim*16)
        # Weight init
        self.apply(init_weights)
        #xavier_uniform_(self.sn_embedding1.weight)

    def forward(self, z, t2i): #, im_emb=None
        # n x 3 x 128 x 128
        h0 = self.opt_block1(z) # n x d_conv_dim   x 64 x 64
        h1 = self.block1(h0)    # n x d_conv_dim*2 x 32 x 32
        h1 = self.self_attn(h1) # n x d_conv_dim*2 x 32 x 32
        h2 = self.block2(h1)    # n x d_conv_dim*4 x 16 x 16
        h3 = self.block3(h2)    # n x d_conv_dim*8 x  8 x  8
        h4 = self.block4(h3)    # n x d_conv_dim*16 x 4 x  4
        h6 = self.block5(h4, downsample=False)  # n x d_conv_dim*16 x 4 x 4
        h6 = self.relu(h6)              # n x d_conv_dim*16 x 4 x 4
        h7 = torch.sum(h6, dim=[2,3])   # n x d_conv_dim*16
        output1 = torch.squeeze(self.snlinear1(h7)) # n x 1
        # Projection
        h_labels = self.snlinear2(t2i)#self.sn_embedding1(labels)   # n x d_conv_dim*16
        proj = torch.mul(h7, h_labels)          # n x d_conv_dim*16
        output2 = torch.sum(proj, dim=[1])      # n x 1
        # Out
        #t2i = t2i.clone().detach() + 0.2*(torch.randn(t2i.size()).cuda())
        #last working begin
        #proj = self.cond_discriminator_mlp( h7, t2i )
        #output2 = self.snlinear2( proj )
        #last working end
        output = output1 + 0.001*output2              # n x 1
        return output



class CondDiscriminatorMLP(nn.Module):
    def __init__(self, txt_dim, im_dim, out_dim, hidden_dim=1024):
        super(CondDiscriminatorMLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.txt_dim = txt_dim
        self.im_dim = im_dim
        self.mlp = snlinear(txt_dim+im_dim, out_dim )#hidden_dim
        self.relu = nn.ReLU()
        #self.out = snlinear(hidden_dim, out_dim)

    def forward(self,x_t,x_im):
        x = torch.cat([x_t,x_im], dim=1)
        x = self.mlp(x)
        x = self.relu(x)
        #x = self.out(x)
        return x 


class Discriminator64(nn.Module):
    """Discriminator."""

    def __init__(self, d_conv_dim, t2i_dim):
        super(Discriminator64, self).__init__()
        self.d_conv_dim = d_conv_dim
        self.opt_block1 = DiscOptBlock(3, d_conv_dim)
        self.block1 = DiscBlock(d_conv_dim, d_conv_dim*2)
        self.self_attn = Self_Attn(d_conv_dim*2)
        self.block2 = DiscBlock(d_conv_dim*2, d_conv_dim*4)
        self.block3 = DiscBlock(d_conv_dim*4, d_conv_dim*8)
        self.block4 = DiscBlock(d_conv_dim*8, d_conv_dim*8)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=d_conv_dim*8, out_features=1)
        self.snlinear2 = snlinear(in_features=d_conv_dim*8, out_features=1)
        #self.sn_embedding1 = sn_embedding(num_classes, d_conv_dim*16)

        self.cond_discriminator_mlp = CondDiscriminatorMLP(txt_dim=t2i_dim, im_dim=d_conv_dim*8, out_dim=d_conv_dim*8)
        # Weight init
        self.apply(init_weights)
        #xavier_uniform_(self.sn_embedding1.weight)

    def forward(self, z, t2i): #, im_emb=None
        # n x 3 x 128 x 128
        h0 = self.opt_block1(z) # n x d_conv_dim   x 32 x 32
        h1 = self.block1(h0)    # n x d_conv_dim*2 x 16 x 16
        h1 = self.self_attn(h1) # n x d_conv_dim*2 x 16 x 16
        h2 = self.block2(h1)    # n x d_conv_dim*4 x 8 x 8
        h3 = self.block3(h2)    # n x d_conv_dim*8 x  4 x  4
        h4 = self.block4(h3, downsample=False)
        h4 = self.relu(h4)              # n x d_conv_dim*8 x 4 x 4
        h5 = torch.sum(h4, dim=[2,3])   # n x d_conv_dim*8
        output1 = torch.squeeze(self.snlinear1(h5)) # n x 1
        # Projection
        #h_labels = self.snlinear2(t2i)#self.sn_embedding1(labels)   # n x d_conv_dim*16
        #proj = torch.mul(h5, h_labels)          # n x d_conv_dim*16
        #output2 = torch.sum(proj, dim=[1])      # n x 1
        # Out


        proj = self.cond_discriminator_mlp( h5, t2i )
        output2 = self.snlinear2( proj )
        output = output1 + output2              # n x 1
        return output 