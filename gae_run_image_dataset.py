from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import scipy.io
from sklearn.metrics import roc_auc_score
from datetime import datetime
import argparse
import os
from model import Dominant
from utils import *
from image_graph_dataset import *
from scipy import ndimage
import sys
from torch_geometric.loader import DataLoader
import numpy as np
import torch.nn as nn
import cv2
from scipy.sparse.linalg import lobpcg
from torch_geometric.utils import get_laplacian, degree
from jw_denoising_filters import *
from utils import scipy_to_torch_sparse, get_operator
import os
import argparse
import torch_geometric
from scipy.fftpack import dct, idct
from numba import jit
import torch.nn.functional as F
import time
from dataset_src.mydataset import *
import torch.distributed as dist
from torch.nn import SyncBatchNorm
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
def get_param_num(net):
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('total params: %d,  trainable params: %d' % (total, trainable))
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding (default: 64)')
parser.add_argument('--ratio_anomal', type=float, default=0.05,
                    help='data number with anomal nodes 150,270,500')
parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
parser.add_argument('--power', type=int, default=1, help='abs or mse loss')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
parser.add_argument('--alpha', type=float, default=0.8, help='balance parameter')
parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
parser.add_argument('--nu', type=float, default=1,
                        help='tight wavelet frame transform tuning parameter')
parser.add_argument('--anomal_num', type=int, default=30,
                    help='number of repetitions')
parser.add_argument('--wd', type=float, default=0.001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=16,
                    help='number of hidden units')
parser.add_argument('--Lev', type=int, default=2,
                    help='level of transform')
parser.add_argument('--s', type=float, default=2,
                    help='dilation scale > 1')
parser.add_argument('--n', type=int, default=4,
                    help='n - 1 = Degree of Chebyshev Polynomial Approximation')
parser.add_argument('--mask', type=int, default=1,
                        help='using mask for local denoising 1/without mask for global denoising 0')
parser.add_argument('--gtmask', type=int, default=0,
                    help='using groundtruth mask mat')
parser.add_argument('--lp', type=int, default=0,
                    help='first term using lp norm')
parser.add_argument('--lq', type=int, default=2,
                    help='second term using lq norm')
parser.add_argument('--boost', type=int, default=0,
                    help='using accelerated scheme or not')
parser.add_argument('--boost_value', type=float, default=0.4,
                    help='boost alue')
parser.add_argument('--stop_thres', type=float, default= 3000000,
                    help='stopping criteria to stop the ADMM')
parser.add_argument('--mu2_0', type=float, default=30,
                    help='initial value of mu2')
parser.add_argument('--anomal_conf_prob', type=float, default=0.1,
                    help='boost alue')
parser.add_argument('--thres_iter', type=float, default=50,
                    help='boost alue')
parser.add_argument('--admm_iter', type=int, default=50,
                    help='number of admm iterations')
parser.add_argument('--rho', type=float, default=0.95,
                    help='piecewise function: constant and > 1')
parser.add_argument('--image_name', type=str, default="034",
                    help='piecewise function: constant and > 1')
parser.add_argument('--mask_stored', type=int, default=1,
                    help='piecewise function: constant and > 1')
parser.add_argument('--localnoise', type=int, default=0,
                    help='local noise or global noise')
parser.add_argument('--train', type=int, default=1,
                    help='local noise or global noise')
parser.add_argument('--local_rank', default=1, type=int,
                        help='node rank for distributed training')
args = parser.parse_args()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
import time
BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_func( mask, labels,power=1):
    # Attribute reconstruction loss
    if power ==1:
        diff_attribute = torch.abs(mask - labels)
        attribute_reconstruction_errors = torch.sum(diff_attribute, 1)
    if power==2:
        diff_attribute = torch.pow(mask - labels,2)
        attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)
    return attribute_cost

def train_dominant(args,normal_dataloaders,train_loader,test_loader,model):
    PATH = "/home/jyh_temp1/Downloads/GraphLocalDenoising/eye_image/"
    dataname = "metal"
    # checkpoint = torch.load(PATH + "eye_model_hid_"+str(args.hidden_dim)+"_normal.pth")
    # model.load_state_dict(checkpoint['net'])
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=3, verbose=True,min_lr=0.000000001)
    min_score = 1000
    loss_list = []
    val_loss_list = []
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        for data in normal_dataloaders:
            x = [tmp_data.x for tmp_data in data]
            x = torch.cat(x).cuda()
            labels = [tmp_data.y for tmp_data in data]
            labels = torch.cat(labels).cuda()
            mask,X_hat = model(data)             ###################input to model##############
            loss = nn.L1Loss()(x,X_hat)####reomve position encoding
            loss.backward()
            optimizer.step()
        loss_list.append(loss.item())
        # if epoch>10:
        #     for data in train_loader:
        #         x = [tmp_data.x for tmp_data in data]
        #         x = torch.cat(x).cuda()
        #         labels = [tmp_data.y for tmp_data in data]
        #         labels = torch.cat(labels).cuda()
        #         mask,X_hat = model(data)             ###################input to model##############
        #         loss = nn.L1Loss()(mask,labels)
        #         loss.backward()
        #         optimizer.step()
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()))
        scheduler.step(loss)
        model.eval()
        val_loss = 0
        for data in test_loader:
            labels = [tmp_data.y for tmp_data in data]
            labels = torch.cat(labels).cuda()
            mask,X_hat = model(data)             ###################input to model##############
            val_loss += nn.L1Loss()(mask,labels).item()
        val_loss = val_loss/len(test_loader)
        val_loss_list.append(val_loss)
        if min_score > val_loss:
            min_score = val_loss
            state = {'net': model.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch + 1,"lr":optimizer.state_dict()['param_groups'][0]['lr']}  ##最好保存lr
            torch.save(state, PATH + "eye_model_hid_"+str(args.hidden_dim)+"_normal_allpos.pth")
    
    X = range(args.epoch)
    plot(X,loss_list, color="red",label="reconstruction losss")
    plot(X,val_loss_list, color="blue",label="anomaly loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    savefig(PATH+"eye_curve.png",dpi = 200)
    print("plot saved!!")
    return X_hat,mask


def test_dominant(args,test_loader,model,out_index=0):
    PATH = "/home/jyh_temp1/Downloads/GraphLocalDenoising/eye_image/"
    checkpoint = torch.load(PATH + "eye_model_hid_"+str(args.hidden_dim)+"_4.2.pth")
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)
    model.eval()
    index = 0
    for data in test_loader:
        data = data.to(device)
        labels = data.y.to(device)
        X_hat = model(data.x, data.edge_index)             ###################input to model##############
        mask = torch.abs(data.x-X_hat)
        val_loss = nn.L1Loss()(mask,labels)
        # print("labels:",mask[100:102,:],labels[100:102,:])
        if index==out_index:
            break ###only get index image
        index+=1
    return X_hat,mask,labels


def torch_vector2image(mask,h,w,ksize,stride,padding):
    mask_img = ((mask.T).unsqueeze(0))
    padding = 0
    mask_img = torch.nn.functional.fold(mask_img,(h, w), kernel_size=(ksize, ksize),stride=(stride,stride),padding=(padding,padding))
    mask_img = mask_img.squeeze()
    mask_img = mask_img.detach().cpu().numpy()
    return mask_img


ksize = 4
stride = 4
padding = 0
h = 440
w = 440
dir_pth = "/home/jyh_temp1/Downloads/GraphLocalDenoising/eye_image/"
y_transforms = transforms.ToTensor()
train_dataset = Anomal_Image_Dataset("/home/jyh_temp1/Downloads/GraphLocalDenoising/eye_image/anomaly/train/image","/home/jyh_temp1/Downloads/GraphLocalDenoising/eye_image/anomaly/train/mask",transform=y_transforms,
                                        target_transform=y_transforms,mode="train")  #####对traindata读取数据，归一化和转为tensor，输出仍为三元组
val_dataset = Anomal_Image_Dataset("/home/jyh_temp1/Downloads/GraphLocalDenoising/eye_image/anomaly/test/image","/home/jyh_temp1/Downloads/GraphLocalDenoising/eye_image/anomaly/test/mask",transform=y_transforms,
                            target_transform=y_transforms,mode="test")
normal_dataset = Normal_Image_Dataset("/home/jyh_temp1/Downloads/GraphLocalDenoising/eye_image/normal/3150FinalRGBBase/",transform=y_transforms,
                                        target_transform=y_transforms,mode="train")
val_dataloaders = DataLoader(val_dataset, batch_size=1)
dataloaders = DataLoader(train_dataset, batch_size=100,num_workers=10)
normal_dataloaders = DataLoader(normal_dataset, batch_size=800,num_workers=10)
model = Dominant(feat_size = 16, hidden_size = args.hidden_dim, dropout = args.dropout).float()
print("num parameter:",get_param_num(model))


###parallel
device = torch.device("cuda", args.local_rank)
dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)
dataloaders = torch_geometric.data.DataListLoader(train_dataset, batch_size=100)
normal_dataloaders = torch_geometric.data.DataListLoader(normal_dataset, batch_size=100)
val_dataloaders = torch_geometric.data.DataListLoader(val_dataset, batch_size=10,shuffle=False,pin_memory=True,drop_last=False)
model = model.to(device)
model = SyncBatchNorm.convert_sync_batchnorm(model)
model = torch_geometric.nn.DataParallel(model, device_ids=[args.local_rank])
print(device)


out_index=1
if args.train==1:
    out,mask = train_dominant(args,normal_dataloaders,dataloaders,val_dataloaders,model)
else:
    dataloaders = DataLoader(train_dataset, batch_size=1)#DataLoader(train_dataset, batch_size=1,num_workers=10)
    out,mask,label = test_dominant(args,dataloaders,model,out_index=out_index)
if args.mask_stored==1:
    mask = torch.load(dir_pth+"mask"+str(args.epoch)+".pkl")
else:
    torch.save(mask.detach().cpu(),dir_pth+"mask"+str(args.epoch)+".pkl")
    mask_img = torch_vector2image(mask,h,w,ksize,stride,padding)
    label_img = torch_vector2image(label,h,w,ksize,stride,padding)
    from matplotlib import pyplot as plt
    #显示差异
    plt.figure() 
    plt.imshow(mask_img, cmap='viridis')  
    plt.colorbar()

    #保存差异图像    
    plt.savefig(dir_pth+"diff"+str(args.epoch)+".png")  
    center = (int(mask_img.shape[1]/2), int(mask_img.shape[0]/2))
    center2 = (88,219)
    # 在图像中心绘制半径为200的白色圆
    circle1 = cv2.circle(np.zeros((440,440)), center, 205, 1, -1)
    circle2 = cv2.circle(np.ones((440,440)), center2, 50, 0, -1)
    mask_img = circle1*mask_img
    mask_img = circle2*mask_img
    mask_img = mask_img/np.max(mask_img)
    thres_list =[0.32,0.34,0.36,0.38]
    
    for thres in thres_list:
        new_mask = np.zeros_like(mask_img)
        new_mask[mask_img>thres]=1
        cv2.imwrite(dir_pth+"mask_"+str(args.epoch)+"thres_"+str(thres)+"out_"+str(out_index)+".png",255*new_mask)
        cv2.imwrite(dir_pth+"label_"+str(args.epoch)+"out_"+str(out_index)+".png",255*label_img)
        from sklearn.metrics import roc_auc_score,average_precision_score
        auc = roc_auc_score(label_img.flatten(),new_mask.flatten(),)
        ap = average_precision_score(label_img.flatten(),new_mask.flatten(),)
        # 输出AUC值
        print("AUC值:", auc,ap,"thres:",thres)

