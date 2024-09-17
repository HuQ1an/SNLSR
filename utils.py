import numpy as np
import scipy.io as sio
import os
import glob
import re
import torch
import torch.nn as nn
import math
import random
def data_augmentation(label, mode=0):
    if mode == 0:
        # original
        return label
    elif mode == 1:
        # flip up and down
        return np.flipud(label)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(label)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(label))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(label, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(label, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(label, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(label, k=3))
        

def prepare_data(path, file_list, file_num,dataset):
    if dataset == 'Chikusei':
        HR_HSI = np.zeros((((512,512,128,file_num))))
        for idx in range(file_num):
            #  read HrHSI
            HR_code = file_list[idx]
            path1 = os.path.join(path) + HR_code + '.mat'
            data = sio.loadmat(path1)
            HR_HSI[:,:,:,idx] = data['gt'] / 15133
        HR_HSI[HR_HSI < 0.] = 0.
        HR_HSI[HR_HSI > 1.] = 1.
    elif dataset == 'Cave':
        HR_HSI = np.zeros((((512,512,31,file_num))))
        for idx in range(file_num):
            #  read HrHSI
            HR_code = file_list[idx]
            path1 = os.path.join(path) + HR_code + '.mat'
            #print(path1)
            data = sio.loadmat(path1)
            HR_HSI[:,:,:,idx] = data['gt']
        HR_HSI[HR_HSI < 0.] = 0.
        HR_HSI[HR_HSI > 1.] = 1.    
    return HR_HSI

def loadpath(pathlistfile):
    fp = open(pathlistfile)
    pathlist = fp.read().splitlines()
    fp.close()
    random.shuffle(pathlist)
    return pathlist











