import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import cv2
from .imsize import imresize
import h5py

class HSTestData(data.Dataset):
    def __init__(self, image_dir, n_scale,dataset):
        self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.n_scale = n_scale
        self.dataset= dataset
    def __getitem__(self, index):
        file_index = index
        load_dir = self.image_files[file_index]

        #data = sio.loadmat(load_dir)
        #img = np.array(data['gt'][...], dtype=np.float32)        
        #gt_size = 512        
        if self.dataset == 'Chikusei':
            data = h5py.File(load_dir,'r')
            img = np.array(data['gt'][...], dtype=np.float32)
            img = img.transpose(2,1,0)
            gt_size = 256
    
            DATAMAX = 15133
            DATAMIN = 0
            img = (img - DATAMIN)/(DATAMAX - DATAMIN)
        elif self.dataset == 'Cave':
            data = sio.loadmat(load_dir)
            img = np.array(data['gt'][...], dtype=np.float32)
            gt_size = 512
         
        #gt = img[440:gt_size_h+440, 392:gt_size_w+392, :]
        gt = img[:,:,:]
        ms = imresize(gt,output_shape = (gt_size // self.n_scale, gt_size // self.n_scale))
        lms = imresize(ms,output_shape =(gt_size, gt_size))
        #print(lms.shape)
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
        #print(lms.shape)
        return ms, lms, gt

    def __len__(self):
        return len(self.image_files)


