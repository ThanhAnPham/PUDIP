#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The shared Net for Phase Unwrapping (YNet)

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
    Ordinary UNet Conv Block (Encoder)
'''
class YNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, activation):
        super(YNetConvBlock, self).__init__()
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
    Ordinary UNet-Up Conv Block including the skip connection (Decoder for segmentation task)
'''
class YNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, activation):
        super(YNetUpBlock, self).__init__()
        self.up         = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        self.bnup       = nn.BatchNorm2d(out_size)
        self.conv       = nn.Conv2d(out_size*2, out_size, kernel_size, stride=1, padding=1)
        self.bn         = nn.BatchNorm2d(out_size)
        self.conv2      = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2        = nn.BatchNorm2d(out_size)
        self.activation = activation


    def forward(self, x, bridge):
        up   = self.up(x)
        up   = self.activation(self.bnup(up))
        #crop1 = self.center_crop(bridge, up.size()[2])
        out  = torch.cat([bridge, up], dim=1)

        out  = self.activation(self.bn(self.conv(out)))
        out  = self.activation(self.bn2(self.conv2(out)))

        return out
    
'''
    Ordinary Autodecoder (Decoder for reconstruction task)
'''
class YNetAuDeBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, activation):
        super(YNetAuDeBlock, self).__init__()
        self.up         = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        self.bnup       = nn.BatchNorm2d(out_size)
        self.conv       = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn         = nn.BatchNorm2d(out_size)
        self.conv2      = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2        = nn.BatchNorm2d(out_size)
        self.activation = activation


    def forward(self, x):
        up   = self.up(x)
        out  = self.activation(self.bnup(up))
        #crop1 = self.center_crop(bridge, up.size()[2])
        

        out  = self.activation(self.bn(self.conv(out)))
        out  = self.activation(self.bn2(self.conv2(out)))

        return out

'''
     YNet consists of the shared Encoder and two individual Decoder for two tasks
'''

class YNet(nn.Module):
    def __init__(self, in_channel, n_classes, convfilt, kernelsize, actifuncrelu):
        super(YNet, self).__init__()
        self.in_channel     = in_channel
        self.n_classes      = n_classes
        self.convfilt       = convfilt
        self.kernelsize     = kernelsize
        self.actifuncrelu   = actifuncrelu
        
        
        
        if self.actifuncrelu:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
    
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        #self.pool5 = nn.MaxPool2d(2, 2)
        
        # Shared Encoder
        self.en_block1 = YNetConvBlock(self.in_channel,self.convfilt[0],self.kernelsize, \
                                               self.activation)
        self.en_block2 = YNetConvBlock(self.convfilt[0],self.convfilt[1],self.kernelsize, \
                                               self.activation)
        self.en_block3 = YNetConvBlock(self.convfilt[1],self.convfilt[2],self.kernelsize, \
                                               self.activation)
        self.en_block4 = YNetConvBlock(self.convfilt[2],self.convfilt[3],self.kernelsize, \
                                               self.activation)
        self.en_block5 = YNetConvBlock(self.convfilt[3],self.convfilt[4],self.kernelsize, \
                                               self.activation)
        
        # Decoder 1 with skip connection (assemble UNet)
        self.up_block1  = YNetUpBlock(self.convfilt[4],self.convfilt[3],self.kernelsize, \
                                             self.activation)
        self.up_block2  = YNetUpBlock(self.convfilt[3],self.convfilt[2],self.kernelsize, \
                                             self.activation)
        self.up_block3  = YNetUpBlock(self.convfilt[2],self.convfilt[1],self.kernelsize, \
                                             self.activation)
        self.up_block4  = YNetUpBlock(self.convfilt[1],self.convfilt[0],self.kernelsize, \
                                             self.activation)
        
        # Decoder 2 without skip connection (assemble AutoEncoder) 
        self.au_block1  = YNetAuDeBlock(self.convfilt[4],self.convfilt[3],self.kernelsize, \
                                             self.activation)
        self.au_block2  = YNetAuDeBlock(self.convfilt[3],self.convfilt[2],self.kernelsize, \
                                             self.activation)
        self.au_block3  = YNetAuDeBlock(self.convfilt[2],self.convfilt[1],self.kernelsize, \
                                             self.activation)
        self.au_block4  = YNetAuDeBlock(self.convfilt[1],self.convfilt[0],self.kernelsize, \
                                             self.activation)
        
        self.conv_last  = nn.Conv2d(self.convfilt[0], self.n_classes, self.kernelsize, stride=1, padding=1)
        #self.last       = nn.Conv2d(self.in_channel, self.n_classes, kernel_size=1, stride=1)
        

    def forward(self, x):
        
        block1    = self.en_block1(x)
        pool1     = self.pool1(block1)

        block2    = self.en_block2(pool1)
        pool2     = self.pool2(block2)

        block3    = self.en_block3(pool2)
        pool3     = self.pool3(block3)

        block4    = self.en_block4(pool3)
        pool4     = self.pool4(block4)
        
        block5    = self.en_block5(pool4)
        
        
        up1       = self.up_block1(block5, block4)
        up2       = self.up_block2(up1, block3)
        up3       = self.up_block3(up2, block2)
        up4       = self.up_block4(up3, block1)
        
        au1       = self.au_block1(block5)
        au2       = self.au_block2(au1)
        au3       = self.au_block3(au2)
        au4       = self.au_block4(au3)
        
        
        out1      = self.conv_last(up4)
        out2      = self.conv_last(au4)
    
        return out1, out2
    
