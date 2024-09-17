import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import cv2
from .imsize import imresize
import h5py

class HSTestDataP(data.Dataset):
    def __init__(self, image_dir, n_scale):
        #self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]、
        
        self.image_files = [os.path.join(image_dir, x) for x in sorted(os.listdir(image_dir),key = lambda x:os.path.getmtime(os.path.join(image_dir,x)))]
        self.n_scale = n_scale

    def __getitem__(self, index):
        file_index = index
        load_dir = self.image_files[file_index]
        #print(load_dir)
        #data = sio.loadmat(load_dir)
        data = h5py.File(load_dir,'r')
        img = np.array(data['gt'][...], dtype=np.float32)

        # 512
        #gt_size = 512    # fixed

        # For ICVL -------------
        '''
        img = np.array(data['rad'][...], dtype=np.float32)
        DATAMAX = np.max(img) 
        DATAMIN = np.min(img)
        img = (img - DATAMIN)/(DATAMAX - DATAMIN) 
        #gt_size_h = 1392
        #gt_size_w = 1296  
        gt_size_h = gt_size  
        gt_size_w = gt_size
        '''
        # For ICVL -------------
        
        # Chikusei 数据集没归一化 在这里归一化
        '''
        DATAMAX = 15133
        DATAMIN = 0
        img = (img - DATAMIN)/(DATAMAX - DATAMIN)
        '''
        channels, width,height = img.shape
        img = img.transpose(2,1,0)
        gt = img[:,:,:]

        #img = img.transpose(1,2,0)
        gt_size_h = 200 #216
        gt_size_w = 160 #224
        #print(gt.shape)
        ms = imresize(gt,output_shape = (gt_size_h // self.n_scale, gt_size_w // self.n_scale))
        #print(ms.shape)
        lms = imresize(ms,output_shape =(gt_size_h, gt_size_w))
        #ms = cv2.resize(gt, (gt_size // self.n_scale, gt_size // self.n_scale), interpolation=cv2.INTER_CUBIC)
  
        #lms = cv2.resize(ms, (gt_size, gt_size), interpolation=cv2.INTER_CUBIC)
        '''
        ms = (ms - ms.min())/(ms.max() - ms.min()) 
        lms = (lms - lms.min())/(lms.max() - lms.min()) 
        '''

        ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
        lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

        ms = ms.type(torch.FloatTensor)
        lms = lms.type(torch.FloatTensor)
        gt = gt.type(torch.FloatTensor)
        ms = torch.clamp(ms,0,1)
        lms = torch.clamp(lms,0,1)
        return ms, lms, gt

    def __len__(self):
        return len(self.image_files)
