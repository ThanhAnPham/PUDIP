#### Phase Unwrapping with Deep Image Prior (PUDIP)

README still under construction.

Relevant publication:

F. Yang, T. -A. Pham, N. Brandenberg, M. P. Lütolf, J. Ma and M. Unser, "Robust Phase Unwrapping via Deep Image Prior for Quantitative Phase Imaging," in IEEE Transactions on Image Processing, vol. 30, pp. 7025-7037, 2021, doi: 10.1109/TIP.2021.3099956.

@ARTICLE{yang2021robust,
  author={Yang, Fangshu and Pham, Thanh-An and Brandenberg, Nathalie and Lütolf, Matthias P. and Ma, Jianwei and Unser, Michael},
  journal={IEEE Transactions on Image Processing}, 
  title={Robust Phase Unwrapping via Deep Image Prior for Quantitative Phase Imaging}, 
  year={2021},
  volume={30},
  number={},
  pages={7025-7037},
  doi={10.1109/TIP.2021.3099956}}

## Requirements

pytorch
matplotlib

Tried on Macbook M1

## Basic Info
PUDIP can be called in Python as a function. Just calling PUDIP without argument will unwrap an example wrapped image.

    from PUDIP import PUDIP
    UnwrapPhase = PUDIP(wrapped_data = None, target = None, parserin = [])

PUDIP can be called as a Python function from command line, e.g.,

    python PUDIP 'example.npy'

Inputs:
wrapped_data is either the filename of the wrapped phase (.npy) or a numpy.ndarray
Same for target (ground-truth)
parserin is either a json filename or a list of dictionaries that overrules the "overlapping" default parameters.
Please have a look at the pars_default.json (and below) for all the possible keys.

Return:
UnwrapPhase: reconstructed phase (size is H*W)

Keys of parser:
## Parameters for data
RealData : true or false (change the way the scalar bias is removed)
ImagSize : Size of wrapped (unwrapped) phase iamge (H*W)
FileName : Name of data (for saved files)
wrapPhase: 2D wrapped phase in the range of [-pi,pi] (size is H*W)

## Parameters for Network
LR           : Learning rate (default:0.01)
NoiseType    : 'Normal' or 'Std': adds noise.normal_()*reg_noise_std or noise.std()*reg_noise_std (default: 'Normal') to input at each iteration
reg_noise_std: Weight for additional noise added to input (default:0.01)
input_num    : Number of input image, 1 for normal and multiples for sequential case (default:1)
input_depth  : Input channel (default:128)
output_depth : Output channel (default:1)
num_iter     : Total number of iterations
reg_loss     : If true will compute the weighted loss: sum(wn.*Unwraploss)
update_ite   : The weights wn will be updated at every update_ite iteration (valid when the reg_loss is true)
boundWeights : the (1/weights) min and max value for the loss weights (see paper)
GDeps        : The constant (GDeps**2) added to loss to make the BP stable (e.g. 1e-9)
Plot         : If true will plot the intermediate results
show_every   : Iteration for intermediate plotting
gpu          : If true will run the code on GPU
gpuID        : GPU ID
main_dir     : Main path (e.g.,'./')

## Parameters for network architecture
convfilt        = [128,128,128,128,128]
kernelsize      = 3
actifuncrelu    = True
LR_decrease     = False        # if Ture will apply the torch.optim.lr_scheduler.StepLR
OptiScStepSize  = 1000         # the parameter is valid when the LR_decrease is True
OptiScGamma     = 0.5          # the parameter is valid when the LR_decrease is True
act_fun         = 'PReLU'      # 'ReLU','LeakyReLU(0.1, inplace=True)','Swish','ELU', 'PReLU'or'none'
INPUT           = 'noise'      # 'noise' or 'meshgrid'
OPTIMIZER       = 'adam'       # 'LBFGS'
OPT_OVER        = 'net'        # 'input', 'net', 'net,input'
NET_TYPE        = 'skip'       # skip, ResNet, UNet,Decoder
upsample_mode   = 'bilinear'     # 'nearest', 'bilinear'; for UNet, deconv is valid
pad             = 'zero'       # 'reflection', 'none','zero'

##             MAIN PARAMETERS (default)

SaveRes   = True   # Save figures and result
ItUpOut   = 100    # Compute Loss, SNR, (save if SaveRes) every ItUpOut iterations
figSize   = 6      # Figure size
BatchSize = 1      # Batch size (only one in DIP)
bgwin     = [3,52,3,52] #Used if RealData is True. Area used to compute the scalar bias (should be a background, no sample-induced phase delay)