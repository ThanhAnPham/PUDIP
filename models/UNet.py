#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:51:19 2019

The Unet for ODT

@author: fangshu.yang@epfl.ch
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
'''
    Initialize the weights
'''

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data, gain = np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias.data, val = 0) 
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            init.constant_(m.weight.data, val = 1.0) 
            init.constant_(m.bias.data, val = 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data, gain = np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias.data, val = 0)
            
        

'''
    Ordinary UNet Conv Block
'''
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, activation):
        super(UNetConvBlock, self).__init__()
        self.conv       = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn         = nn.BatchNorm2d(out_size)
        self.conv2      = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2        = nn.BatchNorm2d(out_size)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out


'''
    Ordinary UNet-Up Conv Block
'''
class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, activation):
        super(UNetUpBlock, self).__init__()
        self.up         = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        self.bnup       = nn.BatchNorm2d(out_size)
        self.conv       = nn.Conv2d(2*out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn         = nn.BatchNorm2d(out_size)
        self.conv2      = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2        = nn.BatchNorm2d(out_size)
        self.activation = activation
    
    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size[2]) // 2
        xy2 = (layer_height - target_size[3]) // 2
        return layer[:, :, xy1:(xy1 + target_size[2]), xy2:(xy2 + target_size[3])]

    def forward(self, x, bridge):
        up   = self.up(x)
        up   = self.activation(self.bnup(up))
        crop1 = self.center_crop(bridge, up.size())
        out  = torch.cat([crop1, up], dim=1)

        out  = self.activation(self.bn(self.conv(out)))
        out  = self.activation(self.bn2(self.conv2(out)))

        return out

'''
     UNet (lateral connection) with long-skip residual connection (from 1st to last layer) (add convolution to the sum of input and output)
'''

class UNet(nn.Module):
    def __init__(self, in_channel, n_classes, convfilt, kernelsize, actifuncrelu):
        super(UNet, self).__init__()
        self.in_channel     = in_channel
        self.n_classes      = n_classes
        self.convfilt       = convfilt
        self.kernelsize     = kernelsize
        self.actifuncrelu   = actifuncrelu
        
        if self.actifuncrelu:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        #self.pool5 = nn.MaxPool2d(2, 2)
        
        
        self.conv_block1_32    = UNetConvBlock(self.in_channel,self.convfilt[0],self.kernelsize, \
                                               self.activation)
        self.conv_block32_64   = UNetConvBlock(self.convfilt[0],self.convfilt[1],self.kernelsize, \
                                               self.activation)
        self.conv_block64_128  = UNetConvBlock(self.convfilt[1],self.convfilt[2],self.kernelsize, \
                                               self.activation)
        self.conv_block128_256 = UNetConvBlock(self.convfilt[2],self.convfilt[3],self.kernelsize, \
                                               self.activation)
        self.conv_block256_512 = UNetConvBlock(self.convfilt[3],self.convfilt[4],self.kernelsize, \
                                               self.activation)
        #self.conv_block512_1024 = UNetConvBlock(self.convfilt[4],self.convfilt[5],self.kernelsize, \
                                               #self.activation,self.bnEps, \
                                               #self.momentum,self.affine,self.track)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        #self.up_block1024_512   = UNetUpBlock(self.convfilt[5],self.convfilt[4],self.kernelsize, \
                                             #self.activation,self.bnEps, \
                                             #self.momentum,self.affine,self.track)
        self.up_block512_256   = UNetUpBlock(self.convfilt[4],self.convfilt[3],self.kernelsize, \
                                             self.activation)
        self.up_block256_128   = UNetUpBlock(self.convfilt[3],self.convfilt[2],self.kernelsize, \
                                             self.activation)
        self.up_block128_64    = UNetUpBlock(self.convfilt[2],self.convfilt[1],self.kernelsize, \
                                             self.activation)
        self.up_block64_32     = UNetUpBlock(self.convfilt[1],self.convfilt[0],self.kernelsize, \
                                             self.activation)
        
        self.conv_last  = nn.Conv2d(self.convfilt[0], self.n_classes, self.kernelsize, stride=1, padding=1)
        self.last       = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1, stride=1)
        

    def forward(self, x):

        block1    = self.conv_block1_32(x)
        pool1     = self.pool1(block1)

        block2    = self.conv_block32_64(pool1)
        pool2     = self.pool2(block2)

        block3    = self.conv_block64_128(pool2)
        pool3     = self.pool3(block3)

        block4    = self.conv_block128_256(pool3)
        pool4     = self.pool4(block4)
        
        block5    = self.conv_block256_512(pool4)
        #pool5     = self.pool5(block5)
        
        #block6    = self.conv_block512_1024(pool5)
        
        #up0       = self.up_block1024_512(block6, block5)
        up1       = self.up_block512_256(block5, block4)
        up2       = self.up_block256_128(up1, block3)
        up3       = self.up_block128_64(up2, block2)
        up4       = self.up_block64_32(up3, block1)
        
        
        last_conv = self.conv_last(up4)
        out       = self.last(last_conv)
       
        return out
    
