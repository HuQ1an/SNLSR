from lzma import CHECK_NONE
import torch
import torch.nn as nn
import numpy as np
import math
import scipy.io as io
import torch.nn.functional as F
def EzConv(in_channel,out_channel,kernel_size):
    return nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=1,padding=kernel_size//2,bias=True)