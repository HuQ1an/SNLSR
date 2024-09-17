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

class HSTrainingDataP(data.Dataset):
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
        
        #data = sio.loadmat(load_dir)
        #似乎mat版本不支持，改为下面这行
        data = h5py.File(load_dir,'r')

        #print(load_dir)
        img = np.array(data['gt'][...], dtype=np.float32)      
        channels, width, height = img.shape
        #print(height)
        #print(width)
        #print(channels)
        img = img.transpose(2,1,0)

        gt_size = 200
        row = random.randint(0, height-gt_size)
        column = random.randint(0, width-gt_size)
        gt = img[row:row+gt_size, column:column+gt_size, :] 
        # For ICVL, the datasize is C H W
        '''
        channels, height, width = img.shape
        img = img.transpose(1,2,0)
        gt_size = 32 * self.n_scale
        row = random.randint(0, height-gt_size)
        column = random.randint(0, width-gt_size)
        gt = img[row:row+gt_size, column:column+gt_size, :] 
        '''
        ms = imresize(gt,output_shape=(gt_size//self.n_scale,gt_size//self.n_scale))
        #print(ms.shape)

        lms = imresize(ms,output_shape=(gt_size,gt_size))
        #print(lms.shape)
        '''
        ms = cv2.resize(gt, (32, 32), interpolation=cv2.INTER_CUBIC)
 
        lms = cv2.resize(ms, (gt_size, gt_size), interpolation=cv2.INTER_CUBIC)
        '''
        
        #ms = (ms - ms.min())/(ms.max() - ms.min())  
        #lms = (lms - lms.min())/(lms.max() - lms.min()) 
        

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
