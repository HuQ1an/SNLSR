import argparse
import os
import sys
import random
import time
import torch
import cv2
import math
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchnet import meter
import utils
import json
from data import HSTrainingData
from data import HSTestData
from data import HSTrainingDataP
from data import HSTestDataP
from data import HSTrainingDataCAVE
from data import HSTestDataCAVE
#from Grad_SR_baseline import SSPSR
from common import *
# loss
from loss import HybridLoss
# from loss import HyLapLoss
from metrics import quality_assessment
import matplotlib.pyplot as plt
import matplotlib

from abundance_SR import SNLSR
# global settings
resume = False
log_interval = 400

model_name = ''
print('Loading '+model_name+' ......')

USE_SSPSR = 0
Parallel = 0

def main():
    # parsers
    main_parser = argparse.ArgumentParser(description="parser for SR network")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_parser.add_argument("--cuda", type=int, required=False,default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    train_parser.add_argument("--batch_size", type=int, default=8, help="batch size, default set to 64")
    train_parser.add_argument("--epochs", type=int, default=60, help="epochs, default set to 20")
    train_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    train_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
    train_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
    train_parser.add_argument("--n_ovls", type=int, default=2, help="n_ovls, default set to 1")
    train_parser.add_argument("--n_scale", type=int, default=8, help="n_scale, default set to 2")
    train_parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")
    train_parser.add_argument("--dataset_name", type=str, default="Cave", help="dataset_name, default set to dataset_name")
    train_parser.add_argument("--model_title", type=str, default="SSPSR", help="model_title, default set to model_title")
    train_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4,
                              help="learning rate, default set to 1e-4")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    train_parser.add_argument("--save_dir", type=str, default="./trained_model/",
                              help="directory for saving trained models, default is trained_model folder")
    train_parser.add_argument("--gpus", type=str, default="1", help="gpu ids (default: 7)")

    test_parser = subparsers.add_parser("test", help="parser for testing arguments")
    test_parser.add_argument("--cuda", type=int, required=False,default=1,
                             help="set it to 1 for running on GPU, 0 for CPU")
    test_parser.add_argument("--gpus", type=str, default="1", help="gpu ids (default: 7)")
    test_parser.add_argument("--batch_size", type=int, default=8, help="batch size, default set to 64")
    test_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    test_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
    test_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
    test_parser.add_argument("--n_ovls", type=int, default=2, help="n_ovls, default set to 1")
    test_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 2")
    test_parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")
    test_parser.add_argument("--dataset_name", type=str, default="Cave", help="dataset_name, default set to dataset_name")
    test_parser.add_argument("--model_title", type=str, default="SSPSR", help="model_title, default set to model_title")
    test_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    # test_parser.add_argument("--test_dir", type=str, required=True, help="directory of testset")
    # test_parser.add_argument("--model_dir", type=str, required=True, help="directory of trained model")

    args = main_parser.parse_args()
    print(args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.subcommand is None:
        print("ERROR: specify either train or test")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    if args.subcommand == "train":
        train(args)
    else:
        test(args)
    pass


def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = start_lr
    lr = start_lr * (0.5 ** (epoch // 700))
    #if epoch>=21 and epoch<=80:
        #lr = start_lr * (0.1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(args, loader, model, criterion):
    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
        for i, (ms, lms, gt) in enumerate(loader):
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            if USE_SSPSR == False: 
                y = model(ms)  
            else:          
                y = model(ms, lms)
            loss = criterion(y, gt)
            epoch_meter.add(loss.item())
        mesg = "===> {}\tEpoch evaluation Complete: Avg. Loss: {:.6f}".format(time.ctime(), epoch_meter.value()[0])
        print(mesg)
    # back to training mode
    model.train()
    return epoch_meter.value()[0]

def test(args):

    if args.dataset_name == 'Cave':
        test_data_dir = '/CAVE/tests/'
    elif args.dataset_name == 'Chikusei':
        test_data_dir = '/Chikusei/testss/'
    elif args.dataset_name == 'Pavia':
        test_data_dir = '/Pavia/tests/' 
    elif args.dataset_name == 'ICVL':
        test_data_dir = '/data2/huqian/ICVL/tests_15/'

    if args.dataset_name=='Cave' or args.dataset_name=='ICVL':
        colors = 31
    elif args.dataset_name=='Pavia':
        colors = 102
    else:
        colors = 128  
    device = torch.device("cuda" if args.cuda else "cpu")
    print('===> Loading testset')
    test_set = HSTestData(test_data_dir,n_scale = args.n_scale,dataset = 'Chikusei')
    #test_set = HSTestDataCAVE(test_data_dir,n_scale = args.n_scale)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    with torch.no_grad():
        epoch_meter = meter.AverageValueMeter()
        epoch_meter.reset()
        # loading model
        model = SNLSR(n_colors=colors,feats=64,factor=4,numend=30)

        # for gpu > 1
        if Parallel:
          model = torch.nn.DataParallel(model)
        
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint["model"].state_dict())
        model.to(device).eval()
        
        mse_loss = torch.nn.MSELoss()
        #output = []
        writer = SummaryWriter('./runs/' + '_' + str(time.ctime()))
      
        test_number = 0

        output_model = '/test_model/'
        output_dataset = '/Chikusei/'
        output_save_path = './outputs' + output_dataset + output_model
        
        for i, (ms, lms, gt) in enumerate(test_loader):
            # compute output
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)

            
            y,rec_input = model(ms, lms) 
            #y = model(ms, lms) 
            # y= model(lms)
            writer.add_image('image/epoch' + str(i) + 'ms', np.expand_dims(ms.cpu().numpy()[0, (15), :, :],axis=0))
            writer.add_image('image/epoch' + str(i) + 'pre', np.expand_dims(y.cpu().detach().numpy()[0, (15), :, :],axis=0))
            writer.add_image('image/epoch' + str(i) + 'lms', np.expand_dims(lms.cpu().numpy()[0, (15), :, :],axis=0))
            writer.add_image('image/epoch' + str(i) + 'gt', np.expand_dims(gt.cpu().numpy()[0, (15), :, :],axis=0))
                    
            #lms, gt = lms.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            #lms = lms[:gt.shape[0],:gt.shape[1],:]
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0],:gt.shape[1],:] 
            
            # Save Output Svg
            
            if not os.path.exists(output_save_path):
                os.makedirs(output_save_path)
            for band in range(colors):
                output_name = 'pic_' + str(i) + '_band_' + str(band) + '.png'
                cv2.imwrite(output_save_path+output_name, y[:,:,band]*255,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                
                
            if i==0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
                print(indices)
                
            else:
                IND = quality_assessment(gt, y, data_range=1., ratio=4)
                print(IND)                
                indices = sum_dict(indices, IND)
            #output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number
    print(test_number)
    print(indices)

    QIstr = str(time.ctime())+ ".txt"
    json.dump(indices, open(QIstr, 'w'))

def save_checkpoint(args, model, epoch):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()
    checkpoint_model_dir = './checkpoints/'
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_filename = args.dataset_name + "_" + args.model_title + "_ckpt_epoch_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    state = {"epoch": epoch, "model": model}
    torch.save(state, ckpt_model_path)
    model.to(device).train()
    print("Checkpoint saved to {}".format(ckpt_model_path))


if __name__ == "__main__":
    main()
