import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import cv2
import utils
import random
from .imsize import imresize
import h5py
import scipy.io as io

class HSTrainingData(data.Dataset):
    def __init__(self, image_dir, n_scale, augment=None):
        self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.augment = augment
        self.n_scale = n_scale
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // 10 //self.factor 
            aug_num = int(index //10 % self.factor)
        load_dir = self.image_files[file_index]
        #print(load_dir)
        
        data = sio.loadmat(load_dir)
        #data = h5py.File(load_dir,'r')

        img = np.array(data['gt'][...], dtype=np.float32)
        
        
        DATAMAX = 15133
        DATAMIN = 0
        img = (img - DATAMIN)/(DATAMAX - DATAMIN)
        
        
        gt_size = 200 
        height, width, channels = img.shape
        
        row = random.randint(0, height-gt_size)
        column = random.randint(0, width-gt_size)
        gt = img[row:row+gt_size, column:column+gt_size, :] 

        ms = imresize(gt,output_shape=(gt_size//self.n_scale,gt_size//self.n_scale))
        lms = imresize(ms,output_shape=(gt_size,gt_size))
        #sprint(ms.shape)

        ms, lms, gt = utils.data_augmentation(ms, mode=aug_num), utils.data_augmentation(lms, mode=aug_num), \
                        utils.data_augmentation(gt, mode=aug_num)


        ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
        lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

        ms = ms.type(torch.FloatTensor)
        lms = lms.type(torch.FloatTensor)
        gt = gt.type(torch.FloatTensor)
        
        ms = torch.clamp(ms,0,1)
        lms = torch.clamp(lms,0,1)
        #print('max'+str(ms.max()))
        #print('min'+str(ms.min()))
        return ms, lms, gt
        
    def __len__(self):
        return len(self.image_files)*self.factor*10
