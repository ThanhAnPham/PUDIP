#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Decoder for Phase Unwrapping 

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
class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, activation):
        super(ConvBlock, self).__init__()
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
    Ordinary Autodecoder
'''
class UpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, activation, upsample_mode):
        super(UpBlock, self).__init__()
        if upsample_mode == 'deconv':
            self.up  = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        elif upsample_mode == 'bilinear':
            self.up     = nn.Upsample(scale_factor=2, mode=upsample_mode)
        elif upsample_mode == 'nearst':
            self.up     = nn.Upsample(scale_factor=2, mode=upsample_mode)
        else:
            assert False
        self.bnup       = nn.BatchNorm2d(out_size)
        self.conv       = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn         = nn.BatchNorm2d(out_size)
        self.conv2      = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2        = nn.BatchNorm2d(out_size)
        self.activation = activation


    def forward(self, x):
        up   = self.up(x)
        out  = self.activation(self.bnup(up))

        out  = self.activation(self.bn(self.conv(out)))
        out  = self.activation(self.bn2(self.conv2(out)))

        return out

'''
     YNet consists of the shared Encoder and two individual Decoder for two tasks
'''

class Decoder(nn.Module):
    def __init__(self, in_channel, n_classes, convfilt, kernelsize, actifuncrelu, upsample_mode):
        super(Decoder, self).__init__()
        self.in_channel     = in_channel
        self.n_classes      = n_classes
        self.convfilt       = convfilt
        self.kernelsize     = kernelsize
        self.actifuncrelu   = actifuncrelu
        self.upsample_mode  = upsample_mode
        
        
        if self.actifuncrelu:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
    
        
        
        # 2 *(Conv+BN+ReLu)
        self.block1 = ConvBlock(self.in_channel,self.convfilt[0],self.kernelsize, \
                                               self.activation)
        self.block2 = ConvBlock(self.convfilt[0],self.convfilt[1],self.kernelsize, \
                                               self.activation)
    
        
        # 4 * (DeConv+BN+ReLu + 2*(Conv+BN+ReLu))
        self.upblock1  = UpBlock(self.convfilt[1],self.convfilt[2],self.kernelsize, \
                                             self.activation,self.upsample_mode)
        self.upblock2  = UpBlock(self.convfilt[2],self.convfilt[3],self.kernelsize, \
                                             self.activation,self.upsample_mode)
        self.upblock3  = UpBlock(self.convfilt[3],self.convfilt[4],self.kernelsize, \
                                             self.activation,self.upsample_mode)
        self.upblock4  = UpBlock(self.convfilt[4],self.convfilt[5],self.kernelsize, \
                                             self.activation,self.upsample_mode)

        # 1* (Conv)
        self.conv_last  = nn.Conv2d(self.convfilt[5], self.n_classes, self.kernelsize, stride=1, padding=1)
        

    def forward(self, x):
        
        block1    = self.block1(x)        
        block2    = self.block2(block1)        
        
        up1       = self.upblock1(block2)
        up2       = self.upblock2(up1)
        up3       = self.upblock3(up2)
        up4       = self.upblock4(up3)    
        
        out      = self.conv_last(up4)
        out      = out[:,:,5:5+150,5:5+100]    
        return out
    
