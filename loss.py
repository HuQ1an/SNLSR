import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]

        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0
        
class spatial_grad(nn.Module):
    def __init__(self,weight):    
        super(spatial_grad, self).__init__()
        self.get_grad = Get_gradient_nopadding()
        self.fidelity = torch.nn.L1Loss()
        self.weight = weight
    
    def forward(self,y,gt):
        y_grad = self.get_grad(y)
        gt_grad = self.get_grad(gt)
        return self.weight * self.fidelity(y_grad,gt_grad)
        
    
class MixLoss(torch.nn.Module):
    def __init__(self):
        super(MixLoss,self).__init__()
        self.fidelity = torch.nn.L1Loss()
        self.grad_loss = spatial_grad(weight=0.5)
        
    def forward(self,y,gt):
        loss = self.fidelity(y, gt)
        loss_grad = self.grad_loss(y,gt)
        return loss+loss_grad




class HybridLoss(torch.nn.Module):
    def __init__(self, lamd=1e-1, spatial_tv=False, spectral_tv=False):
        super(HybridLoss, self).__init__()
        self.lamd = lamd
        self.use_spatial_TV = spatial_tv
        self.use_spectral_TV = spectral_tv
        self.fidelity = torch.nn.L1Loss()
        self.spatial = TVLoss(weight=1e-3)
        self.spectral = TVLossSpectral(weight=1e-3)

    def forward(self, y, gt):
        loss = self.fidelity(y, gt)
        spatial_TV = 0.0
        spectral_TV = 0.0
        if self.use_spatial_TV:
            spatial_TV = self.spatial(y)
        if self.use_spectral_TV:
            spectral_TV = self.spectral(y)
        total_loss = loss + spatial_TV + spectral_TV
        return total_loss

class Aux_loss(torch.nn.Module):
    def __init__(self):
        super(Aux_loss, self).__init__()
        self.L1_loss = torch.nn.L1Loss()
    def forward(self, y_aux, gt):
        loss = 0.0
        for y in y_aux:
            loss = loss + self.L1_loss(y, gt)
        return loss / len(y_aux)


# from https://github.com/jxgu1016/Total_Variation_Loss.pytorch with slight modifications
class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLossSpectral(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLossSpectral, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        # c_tv = torch.abs((x[:, 1:, :, :] - x[:, :c_x - 1, :, :])).sum()
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, :, :]), 2).sum()
        return self.TVLoss_weight * 2 * (c_tv / count_c) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
        
        
        
class hard_loss(torch.nn.Module):        
    def __init__(self,hard_ratio=0.3,bands=31,hard_weight=2.0):
        super(hard_loss,self).__init__()
        self.hard_ratio = hard_ratio
        self.L1_loss = torch.nn.L1Loss()
        self.bands = bands
        self.hard_weight=hard_weight
        
    def forward(self,y,lms,gt):
        batch_size = lms.size()[0]
        spatial_error = abs(gt-lms)
        spatial_error_mean = torch.mean(spatial_error,dim=[2,3])
        sort_result, sort_indice = torch.sort(spatial_error_mean,dim=1,descending=True)
        
        hard_loss = self.L1_loss(y,gt)
        
        for b in range(batch_size):
            hard_indice = sort_indice[b,0:math.ceil(self.bands * self.hard_ratio)]
            batch_loss = 1/batch_size * self.L1_loss(y[b,hard_indice,:,:],gt[b,hard_indice,:,:])
            hard_loss = hard_loss + batch_loss*self.hard_weight
        return hard_loss
        
def cal_gradient_c(x):
  c_x = x.size(1)
  g = x[:, 1:, 1:, 1:] - x[:, :c_x - 1, 1:, 1:]
  return g
  
def cal_gradient_x(x):
  c_x = x.size(2)
  g = x[:, 1:, 1:, 1:] - x[:, 1:, :c_x - 1, 1:]
  return g  
  
def cal_gradient_y(x):
  c_x = x.size(3)
  g = x[:, 1:, 1:, 1:] - x[:, 1:, 1:, :c_x - 1]
  return g

def cal_gradient(inp):
  x = cal_gradient_x(inp)
  y = cal_gradient_y(inp)
  c = cal_gradient_c(inp)
  g = torch.sqrt(torch.pow(x, 2) + torch.pow(y,2) + torch.pow(c,2)+1e-6)
  return g

def cal_sam(Itrue, Ifake):
  esp = 1e-6
  # element-wise product
  # torch.sum(dim=1) 沿通道求和
  # [B C H W] * [B C H W] --> [B C H W]  Itrue*Ifake
  # [B 1 H W] InnerPro(keepdim)
  
  InnerPro = torch.sum(Itrue*Ifake,1,keepdim=True)
  #print('InnerPro')
  #print(InnerPro.shape)
  # 沿通道求范数
  # len1  len2  [B 1 H W] (keepdim)
  len1 = torch.norm(Itrue, p=2,dim=1,keepdim=True)
  len2 = torch.norm(Ifake, p=2,dim=1,keepdim=True)
  #print('len1')
  #print(len1.shape)
  
  divisor = len1*len2
  mask = torch.eq(divisor,0)
  divisor = divisor + (mask.float())*esp 
  cosA = torch.sum(InnerPro/divisor,1).clamp(-1+esp, 1-esp)
  #print(cosA.shape)
  sam = torch.acos(cosA)
  sam = torch.mean(sam) / np.pi
  return sam


def cal_local_sam(Itrue, Ifake,step):
  esp = 1e-6
  # element-wise product
  # torch.sum(dim=1) 沿通道求和
  # [B C H W] * [B C H W] --> [B C H W]  Itrue*Ifake
  # [B 1 H W] InnerPro(keepdim)
  B,C,H,W = Itrue.shape
  
  sam = 0
  sam = torch.tensor(sam).cuda()
  for i in range(C-step):
      local_Itrue = Itrue[:,i:i+step,:,:]
      local_Ifake = Ifake[:,i:i+step,:,:]
      InnerPro = torch.sum(local_Itrue*local_Ifake,1,keepdim=True)
      len1 = torch.norm(local_Itrue, p=2,dim=1,keepdim=True)
      len2 = torch.norm(local_Ifake, p=2,dim=1,keepdim=True)
      divisor = len1*len2
      mask = torch.eq(divisor,0)
      divisor = divisor + (mask.float())*esp 
      cosA = torch.sum(InnerPro/divisor,1).clamp(-1+esp, 1-esp)
      local_sam = torch.acos(cosA)
      local_sam = torch.mean(local_sam.float()) / np.pi
      sam = sam + local_sam
  #print(sam)
  return sam

def cal_sid(Itrue,Ifake):
    # input [B,node,node]
    batch,m,n = Itrue.shape
    sum = 0
    for b in range(batch):
        for r in range(m):
            value = torch.sum(((Itrue[b,r,:]/torch.norm(Itrue,p=1,keepdim=True)) - (Ifake[b,r,:]/torch.norm(Ifake,p=1,keepdim=True))) * (torch.log10(Itrue[b,r,:]/torch.norm(Itrue,p=1,keepdim=True) + 1e-3) - torch.log10(Ifake[b,r,:]/torch.norm(Ifake,p=1,keepdim=True)+ 1e-3)))
            sum = sum+value
    
    sid = sum /(batch*m)
    return sid
            
class HLoss(torch.nn.Module):
    def __init__(self, la1,la2,sam=True, gra=True):
        super(HLoss,self).__init__()
        self.lamd1 = la1
        self.lamd2 = la2
        self.sam = sam
        self.gra = gra
        
        self.fidelity = torch.nn.L1Loss()
        self.gra = torch.nn.L1Loss()
        
    def forward(self, y, gt):
        loss1 = self.fidelity(y, gt)
        loss2 = self.lamd1*cal_sam(y, gt)
        loss3 = self.lamd2*self.gra(cal_gradient(y),cal_gradient(gt))
        loss = loss1+loss2+loss3
        return loss        
        
class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]       

