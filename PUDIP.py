#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 2020 and HEAVILY modified on Tuesday 28 Mar 2023

@author: fangshu.yang@epfl.ch and tampham@mit.edu

Deep Image Prior for phase unwrapping  (Untrained neural network)

            UnwrapPhase = PUDIP(wrapped_data=None, target=None, parserin=None)

wrapped_data is either the filename of the wrapped phase (.npy) or a numpy.ndarray
Same for target
parserin is either a json filename or a list of dictionaries that overrules the "overlapping" default parameters

Return:
UnwrapPhase: reconstructed phase (size is H*W)

"""

from __future__ import print_function
import os, sys, platform

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import numpy as np
import torch

from models.__init__ import get_net

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import json

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False
            
    return params


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_num,input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [input_num, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    
    img_var.detach().cpu().data.numpy()[0]
    '''
    return img_var.detach().cpu().squeeze().numpy()
 
def plot_mat_data(figSize,data,SaveFigPath,FigName):
    """Plot .mat data

    Args: 
        figSize     : size of figure
        data        : data for plotting
        SaveFigPath : path for saving figure
        FigName     : name to save
    """
    fig, ax1 = plt.subplots(1)
    fig.set_figheight(figSize)
    fig.set_figwidth(figSize)

    im1     = ax1.imshow(data)
    divider = make_axes_locatable(ax1)
    cax1    = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1,ax=ax1,cax=cax1)
    #plt.show()
    plt.savefig(SaveFigPath+FigName)
    plt.close(fig)

def unwrap_FD(data):
    """Calculate the finite difference along the horizontal and vertical direction
     
    Args: 
        data        : data 
    """
    dev = data.device
    if len(data.size()) != 4:
        raise Exception('Please reshape the data to correct Dimension!!')
    else: 
        dimData           = data.size()
        
        # Set the boundary as zero
        data_dw       = data[:,:,:,1:] - data[:,:,:,:-1]       
        data_dw_bc    = torch.cat([data_dw,torch.zeros((dimData[0], \
                                                       1,data_dw.size(2),1),device=dev)],dim=3)
        data_dh       = data[:,:,1:,:] - data[:,:,:-1,:]
        data_dh_bc    = torch.cat([data_dh,torch.zeros((dimData[0], \
                                                       1,1,data_dh.size(3)),device=dev)],dim=2)
        data_fd       = torch.cat([data_dw_bc,data_dh_bc],dim=1)
    return data_fd

def unwrap_FD_loss(output_fd, data_fd_mod):      
    """
    Obtain the residual between D(output_fd) and D(W(wrapped_data))
    """
    unwrap_fd_residual = output_fd - data_fd_mod
    unwrap_fd_residual = unwrap_fd_residual
    
    return unwrap_fd_residual

def wrap_formular(data,constant=2*torch.pi):
    """ Calculate the modulo
        Defined as W(data) = mod(data+pi,2*pi)-pi= (data+pi)-floor((data+pi)./(2*pi))*2*pi-pi
        the value range is (-pi,pi)
    """
    #like torch.remainder, but this ways keeps the compatibility with Macbook M1
    wdata = data - torch.div(data+constant/2,constant, rounding_mode="floor") * constant
    return wdata

    
def Plot_Quality(Snr,DataLoss,figSize,SaveFigPath):
    """
       Plot the Loss and SNR
    """
    
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(figSize)
    fig1.set_figwidth(figSize)
            
    line1, = ax1.plot(DataLoss[1:],color='purple',lw=1,ls='-',marker='v',markersize=2,label='DataLoss')
    
    ax1.legend(loc='best',edgecolor='black',fontsize='x-large')
    ax1.grid(linestyle='dashed',linewidth=0.5)
    plt.title('Loss')
    #plt.show()
    plt.savefig(os.path.join(SaveFigPath,'Loss.png'))
    plt.close(fig1)
    
    if Snr is not None:
        fig3, ax3 = plt.subplots()
        fig3.set_figheight(figSize)
        fig3.set_figwidth(figSize)
        plt.plot(Snr[1:])
        plt.title('SNR')
        #plt.show()
        plt.savefig(os.path.join(SaveFigPath,'SNR.png'))

        plt.close(fig3)
    
def Plot_Image(imag,imagc,maxv,minv,i,figSize,SaveFigPath):
    """
       Plot the current recounstructed image
    """
    #imag = imag_np.reshape(imag_np.shape[1],imag_np.shape[2])
    fig1 = plt.figure()
    fig1.set_figheight(figSize)
    fig1.set_figwidth(figSize)
    plt.subplot(1,2,1)
    if maxv == 0 and minv ==0:
        plt.imshow(imag)
    else:
        plt.imshow(imag,vmin=minv,vmax=maxv)
    plt.colorbar()
    plt.title('Result_iteration_'+str(i))
    plt.subplot(1,2,2)
    if maxv ==0 and minv ==0:
        plt.imshow(imagc)
    else:
        plt.imshow(imagc,vmin=minv,vmax=maxv)
    plt.colorbar()
    plt.title('Congruent solution')
    plt.savefig(os.path.join(SaveFigPath,'Result_iteration_{}.png').format(i))
    plt.close(fig1)
        
def SNR(rec,target):
    """
       Calculate the SNR between reconstructed image and true unwrapped image
    """
    if torch.is_tensor(rec):
        rec    = rec.cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    
    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')
    
    TotalSnr = 0.0
    
    if rec.ndim>2 and rec.shape[0] > 1:
        rec      = rec.reshape(rec.shape[0],rec.shape[1],rec.shape[2])
        target   = target.reshape(rec.shape)
     
        for i in range(rec.shape[0]):
            rec_ind     = rec[i,:,:].reshape(np.size(rec[i,:,:]))
            target_ind  = target[i,:,:].reshape(np.size(rec[i,:,:]))
            snr         = 10*np.log10(sum(target_ind**2)/sum((rec_ind-target_ind)**2))
            TotalSnr    = TotalSnr + snr
        TotalSnr = TotalSnr/rec.shape[0]
    else:
        rec       = rec.flatten()
        target    = target.flatten()
        snr       = 10*np.log10(sum(target**2)/sum((rec-target)**2))
        TotalSnr  = snr    
    return TotalSnr

def PUDIP(wrapped_data=None, target=None, parserin=[]):

    with open('params_default.json') as f:
        parser = json.load(f)
    if isinstance(parserin,str): # expects json format
        with open(parserin) as f:
            parserin = json.load(f)
    """else assumes parserin is already a list of dictionaries"""
    
    parser |= parserin
    
    if wrapped_data is None:
        wrapped_data = 'example.npy'
        torch.manual_seed(2020)
    if isinstance(wrapped_data,str):
        wrapped_data = np.float32(np.load(wrapped_data))
    """else assumes wrapped_data is already a numpy array of dim 2 or """
    wrapped_data = np.squeeze(wrapped_data)
    if isinstance(target,str):
        target = np.squeeze(np.float32(np.load(target)))
    elif target is not None:
        target = np.squeeze(target)
    
    if 'FileName' in parser:
        FileName = parser['FileName']
    else:
        FileName = 'generic'
        
    dimData       = wrapped_data.shape
    ImagSize = np.shape(wrapped_data)
    if len(dimData) == 2:
        wrapped_data  = wrapped_data.reshape(-1,1,dimData[0],dimData[1]) # add the number and channel
    
    RealData = parser['RealData']
    LR  = parser['LR']
    NoiseType = parser['NoiseType']
    reg_noise_std = parser['reg_noise_std']
    input_num  = parser['input_num']
    input_depth = parser['input_depth']
    output_depth = parser['output_depth']
    num_iter = parser['num_iter']
    reg_loss = parser['reg_loss']
    update_ite = parser['update_ite']
    boundWeights = parser['boundWeights']
    GDeps = parser['GDeps']
    SaveRes = parser['SaveRes']
    show_every = parser['show_every']
    gpu = parser['gpu']
    gpuID = parser['gpuID']
    if 'main_dir' in parser:
        main_dir = parser['main_dir']    
    else:
        main_dir = '.' + os.sep
    ItUpOut = parser['ItUpOut']
    
    INPUT = parser["INPUT"]
    OPTIMIZER = parser["OPTIMIZER"]
    OPT_OVER = parser["OPT_OVER"]
    NET_TYPE = parser["NET_TYPE"]
    LR_decrease =parser["LR_decrease"]
    OptiScStepSize =parser["OptiScStepSize"]
    OptiScGamma = parser["OptiScGamma"]
    pad = parser["pad"]    
    convfilt = parser["convfilt"]
    figSize = parser["figSize"]
    upsample_mode = parser["upsample_mode"]
    act_fun = parser["act_fun"]
    BatchSize = parser["BatchSize"]
    
    sys.path.append(os.getcwd())

    if RealData:
        tag0 = 'Real_'+str(FileName)+'_PhaseUnwrap_DIP_'
    else:    
        tag0 = 'Simulate_'+str(FileName)+'_PhaseUnwrap_DIP_'
           
    tag1 = 'NET_' + str(NET_TYPE)
    tag2 = '_Optimizer_' + str(OPTIMIZER)
    tag3 = '_Pad_' +str(pad)
    tag4 = '_LR_' + str(LR)
    if LR_decrease:
        tag44 = '_Step_' + str(OptiScStepSize) + '_Ratio_' + str(OptiScGamma)
    else:
        tag44 = ''
    tag5 = '_InputDepth_' + str(input_depth)
    tag6 = '_OutputDepth_' + str(output_depth)
    tag7 = '_NumIte_' + str(num_iter)
    if reg_noise_std>0:
        tag8 = '_NoiseType_' + str(NoiseType) +'_RegNoise_' +str(reg_noise_std)
    else:
        tag8 = ''
    tag10 = '_ConvFit_' + str(convfilt)
    tag11 = '_RegData_' + str(reg_loss)
    if reg_loss:
        tag12 = '_RegIte_' + str(update_ite) + '_BoundWeight_' + str(boundWeights)
    else:
        tag12 = ''
    tag13 = '_GDeps_' + str(GDeps)

    ResultFileName = tag0+tag1+tag4+tag44+tag5+tag6+tag7+tag8+tag11+tag12
    NetworkName    = tag0+tag1+tag4+tag44+tag5+tag6+tag7+tag8+tag11+tag12

    #####################################################
    #####                  PATHS                    #####
    #####################################################
    # Main path
    if len(main_dir) == 0:
        raise Exception('Please specify path to correct directory!!')


    # Result and Model path for phase unwrapping
    if os.path.exists('PUresults'): #and os.path.exists('PUnetworks'):
        results_dir      = main_dir + 'PUresults' + os.sep 
        #networks_dir     = main_dir + 'PUnetworks' + os.sep
    else:
        os.makedirs('PUresults')
        #os.makedirs('PUnetworks')
        results_dir      = main_dir + 'PUresults' + os.sep
        #networks_dir     = main_dir + 'PUnetworks' + os.sep
    ResultPath = results_dir + ResultFileName
    #NetworkPath = networks_dir + NetworkName

    if os.path.exists(ResultPath):    
        ResultPath = ResultPath    
    else:
        os.makedirs(ResultPath)    
        ResultPath = ResultPath


    """Check CUDNN and GPU"""
    if gpu:
        
        computer = platform.system()
        if computer=='Darwin': #mac
            print('On Mac')
            device = torch.device("mps")
        else:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark =True
            device = torch.device('cuda:{}'.format(gpuID) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device("cpu")
        
    """Plot measurement"""
    if SaveRes:
        if wrapped_data.shape[0] == 1:
            imag  = wrapped_data.reshape(wrapped_data.shape[2],wrapped_data.shape[3])
        FigName = os.sep+'wrapPhase.png'
        plot_mat_data(figSize=figSize,data=imag,SaveFigPath=ResultPath,FigName=FigName)


    """Define the network"""
    net = get_net(input_depth=input_depth,NET_TYPE=NET_TYPE, \
                  upsample_mode=upsample_mode,pad=pad, \
                  n_channels=output_depth,skip_n33d=128, \
                  skip_n33u=128,skip_n11=4,num_scales=5,act_fun=act_fun)
    net = net.to(device)

    """Get input:Returns a pytorch.Tensor of size (input_num x `input_depth` x `ImagSize[0]` x `ImagSize[1]`) """
   
    net_input = get_noise(input_num=1,input_depth=input_depth, \
                          method=INPUT,spatial_size=ImagSize, \
                          noise_type='u',var=1./10).to(device).detach() 
    
    net_input = net_input.to(device)

    """Convert the numpy array to torch tensor"""
    wrapped_data   = torch.from_numpy(wrapped_data)
    wrapped_data   = wrapped_data.to(device)

    """Compute number of parameters"""
    s  = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)


    print() 
    print('*******************************************') 
    print('*******************************************') 
    print('                STARTING                   ') 
    print('*******************************************') 
    print('*******************************************') 
    print() 
    #################### Init globals ################# 
    net_input_saved  = net_input.detach().clone()
    noise            = net_input.detach().clone()
    i                = 1
    outIte           = 0
    
    if target is not None:
        Snr            = []
    else:
        Snr            = None
    DataLoss          = 0.0
    
    #################### Compute the W(D(wrapped_data)) ################# 
    data_fd      = unwrap_FD(wrapped_data)
    data_fd_mod  = wrap_formular(data_fd)  


    #####################  Run Code ####################
    t0  = time.time()
    p      = get_params(OPT_OVER, net, net_input)

    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(p, lr=LR)
    if LR_decrease:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                                    step_size=OptiScStepSize, \
                                                    gamma=OptiScGamma)
    for j in range(num_iter):
        if LR_decrease:
            scheduler.step()
        
        optimizer.zero_grad()
        """ Update input random variant in each iteration"""            
        if reg_noise_std > 0:
            if NoiseType == 'Std':
                net_input = net_input_saved + (noise.std() * reg_noise_std) 
            elif NoiseType == 'Normal':
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            elif NoiseType == 'None':    
                net_input = net_input_saved
            else:
                assert False

        """Obtain output of network"""
        if output_depth == 1:
            output_unwrap = net(net_input)
        else:
            assert False

        """Remove the (scalar) bias"""
        if RealData:            
            output_unwrap = output_unwrap - output_unwrap[0,0].narrow(0,bgwin[0],bgwin[1]).narrow(1,bgwin[2],bgwin[3]).mean().clone() #torch.mean(output_unwrap[0,0,3:53,3:53].clone())
        else:
            output_unwrap = output_unwrap - torch.min(output_unwrap.clone())
        

        """Obtain the finite difference operation"""
        output_fd = unwrap_FD(data=output_unwrap)
        unwrap_fd_res  = unwrap_FD_loss(output_fd, data_fd_mod)

        """Add small constant eps**2 for the gradient numerical stability"""
        fd_squareloss = torch.pow(unwrap_fd_res[:,0,:,:].view(BatchSize,1,ImagSize[0],ImagSize[1]),2) \
        + torch.pow(unwrap_fd_res[:,1,:,:].view(BatchSize,1,ImagSize[0],ImagSize[1]),2)

        fd_squareloss = fd_squareloss + GDeps**2
        """Initialize the weights as 1"""
        wn = torch.ones_like(output_unwrap).detach().clone()

        """if reg_loss is True will get the weighted loss, the weights will be updated every update_ite iteration"""
        if reg_loss:
            if i < update_ite+1:        
                wn = torch.ones_like(output_unwrap).detach().clone()
            elif i > outIte*update_ite and i < (outIte+1)*update_ite+1:

                epsn = torch.sqrt(fd_squareloss)
                epsn = epsn.detach().clone()

                wn =  1./(torch.max(torch.min(epsn, boundWeights[1]*torch.ones_like(epsn).detach().clone()),
                                    boundWeights[0]*torch.ones_like(epsn).detach().clone()))
            else:
                assert False

        """the data loss is the sqrt of the weighted difference between D(output) and W(D(data))"""

        data_loss = torch.sum(torch.mul(wn,(torch.pow(fd_squareloss,0.5)).view(output_unwrap.size())))
        
        """Backpropagate loss""" 
        data_loss.backward()


        """Record Loss"""
        DataLoss    = np.append(DataLoss, data_loss.item())  
        rec = torch_to_np(output_unwrap+wrap_formular(wrapped_data-output_unwrap))
        
        """History"""
        if i % ItUpOut==0:
            print ('Ite: %05d  Data Loss: %f' % (i,data_loss.item())) 

        """Plot current result, Loss and SNR """
        if SaveRes and i % show_every == 0:
            with torch.no_grad():
                if target is not None:
                    Snr.append(SNR(rec,target))
                Plot_Quality(Snr=Snr,DataLoss=DataLoss, \
                                figSize=figSize,SaveFigPath=ResultPath)
                #output_Unwrap_np = torch_to_np(output_unwrap)
                output_Unwrap_np_c = torch_to_np(output_unwrap + wrap_formular(wrapped_data - output_unwrap))
                Plot_Image(torch_to_np(output_unwrap),output_Unwrap_np_c,maxv=0, \
                            minv=0,i=i,figSize=figSize, \
                            SaveFigPath=ResultPath)

        """Update the iteration"""
        if i % update_ite == 0:
            outIte += 1

        i += 1

        optimizer.step()

    ###################  Final Output #####################
    
    out  = net(net_input)
    """Remove a scalar bias"""
    if RealData:
        out = out - out[0,0].narrow(0,bgwin[0],bgwin[1]).narrow(1,bgwin[2],bgwin[3]).mean().clone() #torch.mean(out[0,0,3:53,3:53].clone())
    else:
        out = out - out.min()
    """Make the solution congruent with the measurements"""
    out_np  = torch_to_np(out + wrap_formular(wrapped_data - out)).reshape(ImagSize[0],ImagSize[1])
    
    if SaveRes:
        np.save(os.path.join(ResultPath,'unwrapped.npy'), out_np)
    Plot_Image(torch_to_np(out),out_np, maxv=0, \
                            minv=0,i=i+1,figSize=figSize, \
                            SaveFigPath=ResultPath)

    """Record the total time"""
    time_elapsed = time.time() - t0
    print()
    print('Total Time:  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return out_np

if __name__=="__main__":
    args = sys.argv[1:]
    PUDIP(*args)