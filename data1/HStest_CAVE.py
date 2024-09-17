import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import cv2
from .imsize import imresize
import h5py

class HSTestDataCAVE(data.Dataset):
    def __init__(self, image_dir, n_scale):
        self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.n_scale = n_scale

    def __getitem__(self, index):
        file_index = index
        load_dir = self.image_files[file_index]
        #print(load_dir)
        data = sio.loadmat(load_dir)
        #data = h5py.File(load_dir,'r')
        img = np.array(data['gt'][...], dtype=np.float32)

        # 512
        gt_size = 512    # fixed
        # Chikusei 归一化
        '''
        DATAMAX = 15133
        DATAMIN = 0
        img = (img - DATAMIN)/(DATAMAX - DATAMIN)
        '''
        gt = img[0:gt_size, 0:gt_size, :]
        ms = imresize(gt,output_shape = (gt_size // self.n_scale, gt_size // self.n_scale))
        lms = imresize(ms,output_shape =(gt_size, gt_size))

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