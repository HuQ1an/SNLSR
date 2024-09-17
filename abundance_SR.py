import torch
import torch.nn as nn
import numpy as np
import math
import scipy.io as io
import torch.nn.functional as F
import layers
def EzConv(in_channel,out_channel,kernel_size):
    return nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=1,padding=kernel_size//2,bias=True)

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out = out * x
        return out

class SANL(nn.Module): 
    def __init__(self,n_feat):
        super(SANL,self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(n_feat,n_feat,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat,n_feat,kernel_size=3,stride=1,padding=1),
        )
        self.convq = nn.Sequential(
            nn.Linear(n_feat,n_feat,bias=False),
        )
        self.convk = nn.Sequential(
            nn.Linear(n_feat,n_feat,bias=False),
        )
        self.convv = nn.Sequential(
            nn.Linear(n_feat,n_feat,bias=False),
        )
        self.to_out = nn.Linear(n_feat,n_feat)
        
    def forward(self,x):
        x = self.embedding(x)
        
        # input:[B,L,H,W], L:Num of Spectrals
        B, L, H, W = x.size()
        # x_re:[B,HW,l]
        x_re = x.view(B,L,H*W).permute(0,2,1)
        
        x_emb1 = self.convq(x_re) 
        x_emb2 = self.convk(x_re)
        x_emb3 = self.convv(x_re)
        
        x_emb1 = F.normalize(x_emb1, dim=-1, p=2)
        x_emb2 = F.normalize(x_emb2, dim=-1, p=2)

        x_emb1 = torch.unsqueeze(x_emb1,dim=3)
        x_emb2 = torch.unsqueeze(x_emb2,dim=2)

        mat_product = torch.matmul(x_emb1,x_emb2)
        mat_product = F.softmax(mat_product, dim=3)

        x_emb3 = torch.unsqueeze(x_emb3,dim=3)

        attention = mat_product     

        out = torch.matmul(attention,x_emb3)
        out = torch.squeeze(out,dim=3)
        out = self.to_out(out)

        out = out.permute(0,2,1).view(B, L, H, W)
        return out + x   
    
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m

class DDP(nn.Module):
    def __init__(self,n_feats):
        super(DDP, self).__init__()
        self.pointconv = nn.Conv2d(n_feats,n_feats,1)
        self.depthconv = nn.Conv2d(n_feats,n_feats,kernel_size=3,padding=1,groups=n_feats)
        
    def forward(self,x,spex,spax):
        diffspex = self.pointconv(spex - x) 
        diffspax = self.depthconv(spax - x)        
        return x + diffspex + diffspax

class SSAM(nn.Module):  
    def __init__(self,n_feats,head,num):
        super(SSAM, self).__init__()
        self.ESA = ESA(n_feats,nn.Conv2d)
        self.NL = SANL(n_feats)
        self.head = head
        if head==0:
            self.botnek = nn.Conv2d(n_feats*num,n_feats,kernel_size=1)
        self.DDP = DDP(n_feats)
        
    def forward(self,x):
        if self.head==0:
            x = self.botnek(x)
        # HSR15 ����res
        spex = self.NL(x)
        spax = self.ESA(x)
        out = self.DDP(x,spex,spax)      
        return out
        
class SSAMC(nn.Module):  
    def __init__(self,n_feats,head,num):
        super(SSAMC, self).__init__()
        self.ESA = ESA(n_feats,nn.Conv2d)
        self.NL = SANL(n_feats)
        self.head = head
        if head==0:
            self.botnek = nn.Conv2d(n_feats*num,n_feats,kernel_size=1)
        self.fusion = nn.Conv2d(n_feats*2,n_feats,kernel_size=1)
        
    def forward(self,x):
        if self.head==0:
            x = self.botnek(x)
        spex = self.NL(x)
        spax = self.ESA(x)
        
        out = self.fusion(torch.cat([spax,spex],dim=1)) 
        return out + x
        

class Upsample(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True, conv=EzConv):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsample, self).__init__(*m)

      
class SNLSR(nn.Module):  
    def __init__(self,n_colors,feats,factor,numend):
        super(SNLSR, self).__init__()
        self.act = torch.nn.Tanh()
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.SPDN = nn.Sequential(
            nn.Conv2d(n_colors,8*numend,kernel_size=3,padding=1),
            SA(),
            torch.nn.Tanh(),
            nn.Conv2d(8*numend,4*numend,kernel_size=3,padding=1),
            SA(),
            torch.nn.Tanh(),
            nn.Conv2d(4*numend,2*numend,kernel_size=3,padding=1),
            SA(),
            torch.nn.Tanh(),    
            nn.Conv2d(2*numend,1*numend,kernel_size=1), 
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)               
        )
        self.factor = factor
        self.SRhead = nn.Conv2d(numend,feats,kernel_size=3,padding=1)
        
        self.block1 = SSAM(feats,1,1)
        self.block2 = SSAM(feats,0,2)
        self.block3 = SSAM(feats,0,3)

        self.block4 = SSAM(feats,1,1)
        self.block5 = SSAM(feats,0,2)
        self.block6 = SSAM(feats,0,3)

        
        self.Up = nn.Sequential(
            nn.Conv2d(feats*4,feats,kernel_size=1),
            Upsample(scale=factor//2, n_feats=feats),
        )
        self.Up2 = nn.Sequential(
            nn.Conv2d(feats*4,feats,kernel_size=1),
            Upsample(scale=factor//2, n_feats=feats),
            nn.Conv2d(feats,numend,kernel_size=3,padding=1)
        )   
        self.endmember = nn.Conv2d(numend,n_colors,kernel_size=1,bias=False)
        
    def forward(self,ms,lms):
        abu = self.SPDN(ms)
        rec_input = self.endmember(abu)
        abu = self.SRhead(abu)
        abu1 = self.block1(abu)
        abu2 = self.block2(torch.cat([abu,abu1],dim=1))
        abu3 = self.block3(torch.cat([abu,abu1,abu2],dim=1))
   
        tempAbu = self.Up(torch.cat([abu,abu1,abu2,abu3],dim=1))
        
        abu4 = self.block4(tempAbu)
        abu5 = self.block5(torch.cat([tempAbu,abu4],dim=1))
        abu6 = self.block6(torch.cat([tempAbu,abu4,abu5],dim=1))

        Abu = self.Up2(torch.cat([tempAbu,abu4,abu5,abu6],dim=1))
        SR = self.endmember(Abu) + lms
        return SR,rec_input 

