import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir)
print("current dir:",current_dir)
from torch.utils.data import Dataset
import PIL.Image as Image
import torch
import math
# from radiomics import featureextractor
# import SimpleITK as sitk
import time
# import six
import random
import scipy
from torch_geometric.data import InMemoryDataset, download_url, Data,Dataset
from torchvision import transforms
import torchvision.transforms.functional as f
from scipy import ndimage
from multiprocessing import Pool
from p_tqdm import p_map
# import stainNorm_Reinhard
from torch_geometric.utils.augmentation import *
import torch
import glob
import cv2
import scipy.sparse as sp
import numpy as np
def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    #print("pading:",paddings)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings

def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             padding=0,
                             stride=strides)
    # print("sha:",images.shape)
    patches = unfold(images)
    return patches, paddings

def local_adj(patch,row_num,col_num):
    num_nodes = row_num*col_num
    adj = np.zeros((num_nodes,num_nodes))
    ##consider self loop
    for k in range(num_nodes):
        for j in range(k,num_nodes):
            if k == j:
                adj[k, j] = 1
            if k - j == 1 or j - 1 == 1:
                adj[k, j] = 1  # The gauss kernel can be applied
            if k - j == col_num or j - k == col_num:
                adj[k, j] = 1
    adj = adj + adj.T
    return adj + np.eye(adj.shape[0])

def nonlocal_adj(patch, row_num, col_num,thres=35000):
    num_nodes = row_num * col_num
    adj = np.zeros((num_nodes, num_nodes))
    ##consider self loop
    for i in range(num_nodes):
        for j in range(i,num_nodes):
            patch1 = patch[i,:]
            patch2 = patch[j,:]
            dct1 = dct(patch1)
            dct2 = dct(patch2)
            #print("dct of node i&j:", dct1,dct2)
            tmp = np.sum(np.abs(dct1-dct2))
            # print("dct of node i&j:", i,j,tmp)
            #print("tmp value of ",str(i),"and",str(j),tmp)
            if tmp< thres:
                adj[i,j]=1
    adj = adj + adj.T
    return adj


class Anomal_Image_Dataset(Dataset):
    def __init__(self, img_dir, ref_dir,transform=None, target_transform=None,mode="train"):

        self.npath = img_dir  # os.path.join(img_dir, "noisy")
        self.rpath = ref_dir  # os.path.join(img_dir, "ref")
        self.nimg_name = sorted(os.listdir(self.npath))
        self.rimg_name = sorted(os.listdir(self.rpath))
        self.nimg_name = [
            i
            for i in self.nimg_name
            if i.split(".")[-1].lower() in ["jpeg", "jpg", "png", "bmp", "tif"]
        ]
        self.rimg_name = [
            i
            for i in self.rimg_name
            if i.split(".")[-1].lower() in ["jpeg", "jpg", "png", "bmp", "tif"]
        ]
        self.transform = transform
        self.target_transform = target_transform
        self.pth = "/home/jyh_temp1/Downloads/GraphLocalDenoising/eye_image/anomaly/"+mode +"/graph/"
        ###"iter"+str(self.iter_num)+"compact"+str(self.compact)+
        if os.path.exists(self.pth) and len(os.listdir(self.pth))>1:
            pass
        else:
            os.makedirs(self.pth,exist_ok=True)
            self.preprocess_graph()
        super(Anomal_Image_Dataset, self).__init__()
    def len(self):
        return len(self.rimg_name)
    def save(self,idx):
        ksize = 4
        stride = 4
        padding = 0
        tmp_train_name = self.nimg_name[idx]
        nimg_name = os.path.join(self.npath, tmp_train_name)
        rimg_name = os.path.join(self.rpath, tmp_train_name)  # self.rimg_name[idx])
        img_x = Image.open(nimg_name).convert('L')  ############读取路径指向图片，非numpy类型
        img_y = Image.open(rimg_name).convert('L')
        ######empoly SLIC get superpxl
        origin_img = cv2.resize(np.array(img_x),(440,440))
        gt_mask_image = cv2.resize(np.array(img_y),(440,440),interpolation= cv2.INTER_NEAREST)
        old_mask_image = gt_mask_image/np.max(gt_mask_image)
        gt_mask_image = np.zeros_like(old_mask_image)
        gt_mask_image[old_mask_image>0.5]=1
        h = origin_img.shape[0]
        w = origin_img.shape[1]
        row_num = h/stride
        col_num = w/stride
        num_nodes = int(row_num*col_num)
        img = torch.from_numpy(origin_img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img.float()
        gt_mask = torch.from_numpy(gt_mask_image)
        gt_mask = gt_mask.unsqueeze(0)
        gt_mask = gt_mask.unsqueeze(0)
        gt_mask = gt_mask.float()
        unfold = torch.nn.Unfold(kernel_size=ksize,stride=stride,padding=padding)
        #print("img:",img,img.shape)
        patches = unfold(img)
        patches = (patches.squeeze(0)).squeeze(0)
        feat_mat = (patches.T)#####num_nodes*size 
        ##mask
        gt_patches = unfold(gt_mask)
        gt_patches = (gt_patches.squeeze(0)).squeeze(0)
        gt_mask_node = gt_patches.T
        ###adj construction
        adj1 = local_adj(feat_mat,int(row_num),int(col_num))
        '''nonlocal adjacency'''
        start = time.time()
        adj2 = 0#nonlocal_adj(feat_mat,int(row_num),int(col_num),thres=8000)
        end = time.time()
        print("adjacency completed, cost time:",end-start)
        adj = adj1 + adj2
        adj[adj>1]=1
        adj = sp.coo_matrix(adj)
        values = adj.data  
        indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
        adj = torch.LongTensor(indices)  # PyG框架需要的coo形式
        ##instance edge is binary label that only connects superpixels within one nuclie 
        data = Data(x=feat_mat,edge_index=adj, y=gt_mask_node,img=torch.from_numpy(origin_img),img_y=torch.from_numpy(gt_mask_image))
        torch.save(data,self.pth+str(idx)+".pth")
        print("saved for "+str(idx))
    def preprocess_graph(self):
        # self.mean,self.std = self.get_all_sample_mean()
        p_map(self.save,range(len(self.rimg_name)))#
    def get(self, idx):
        data = torch.load(self.pth+str(idx)+".pth")
        # x = data.x 
        # pos = torch.arange(0, 110*110) ##pos should be the same size with feat matrix
        # pos = pos.reshape(110*110,1)
        # x = torch.cat([pos,x],dim=1)
        # data.x = x

        ##add postion 
        data.feat_num = data.x.shape[1]
        position = torch.arange(0, data.x.shape[0]).unsqueeze(1)
        # create matrix of divisors (powers of 10000) with exponents ranging from 0 to d_model-1
        div_term = torch.exp(torch.arange(0, data.feat_num, 2) * -(math.log(10000.0) / data.feat_num)) ##data.feat_num must be even
        # create a matrix by applying div_term elementwise to the positional matrix and taking
        # the sine and cosine of each odd and even element, respectively
        pe = torch.zeros_like(data.x)
        pe[:, 0::2] = torch.sin(position * div_term) ##even
        pe[:, 1::2] = torch.cos(position * div_term)  ##odd
        data.x = data.x+pe
        return data



class Normal_Image_Dataset(Dataset):
    def __init__(self, img_dir,transform=None, target_transform=None,mode="train"):

        self.npath = img_dir  # os.path.join(img_dir, "noisy")
        self.nimg_name = sorted(os.listdir(self.npath))
        self.nimg_name = [
            i
            for i in self.nimg_name
            if i.split(".")[-1].lower() in ["jpeg", "jpg", "png", "bmp", "tif"]
        ]
        self.transform = transform
        self.target_transform = target_transform
        self.pth = "/home/jyh_temp1/Downloads/GraphLocalDenoising/eye_image/normal/"+mode +"/graph/"
        ###"iter"+str(self.iter_num)+"compact"+str(self.compact)+
        if os.path.exists(self.pth) and len(os.listdir(self.pth))>1:
            pass
        else:
            os.makedirs(self.pth,exist_ok=True)
            self.preprocess_graph()
        super(Normal_Image_Dataset, self).__init__()
    def len(self):
        return len(self.nimg_name)
    def save(self,idx):
        ksize = 4
        stride = 4
        padding = 0
        tmp_train_name = self.nimg_name[idx]
        nimg_name = os.path.join(self.npath, tmp_train_name)
        img_x = Image.open(nimg_name).convert('L')  ############读取路径指向图片，非numpy类型
        ######empoly SLIC get superpxl
        origin_img = cv2.resize(np.array(img_x),(440,440))
        gt_mask_image = np.zeros_like(origin_img)
        h = origin_img.shape[0]
        w = origin_img.shape[1]
        row_num = h/stride
        col_num = w/stride
        num_nodes = int(row_num*col_num)
        img = torch.from_numpy(origin_img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img.float()
        gt_mask = torch.from_numpy(gt_mask_image)
        gt_mask = gt_mask.unsqueeze(0)
        gt_mask = gt_mask.unsqueeze(0)
        gt_mask = gt_mask.float()
        unfold = torch.nn.Unfold(kernel_size=ksize,stride=stride,padding=padding)
        #print("img:",img,img.shape)
        patches = unfold(img)
        patches = (patches.squeeze(0)).squeeze(0)
        feat_mat = (patches.T)#####num_nodes*size 
        ##mask
        gt_patches = unfold(gt_mask)
        gt_patches = (gt_patches.squeeze(0)).squeeze(0)
        gt_mask_node = 1-(gt_patches.T)
        ###adj construction
        adj1 = local_adj(feat_mat,int(row_num),int(col_num))
        '''nonlocal adjacency'''
        start = time.time()
        adj2 = 0#nonlocal_adj(feat_mat,int(row_num),int(col_num),thres=8000)
        end = time.time()
        print("adjacency completed, cost time:",end-start)
        adj = adj1 + adj2
        adj[adj>1]=1
        adj = sp.coo_matrix(adj)
        values = adj.data  
        indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
        adj = torch.LongTensor(indices)  # PyG框架需要的coo形式
        ##instance edge is binary label that only connects superpixels within one nuclie 
        data = Data(x=feat_mat,edge_index=adj, y=gt_mask_node,img=torch.from_numpy(origin_img),img_y=torch.from_numpy(gt_mask_image))
        torch.save(data,self.pth+str(idx)+".pth")
        print("saved for "+str(idx))
    def preprocess_graph(self):
        # self.mean,self.std = self.get_all_sample_mean()
        p_map(self.save,range(len(self.nimg_name)))#
    def get(self, idx):
        data = torch.load(self.pth+str(idx)+".pth")
        ##add postion 
        data.feat_num = data.x.shape[1]
        position = torch.arange(0, data.x.shape[0]).unsqueeze(1)
        # create matrix of divisors (powers of 10000) with exponents ranging from 0 to d_model-1
        div_term = torch.exp(torch.arange(0, data.feat_num, 2) * -(math.log(10000.0) / data.feat_num)) ##data.feat_num must be even
        # create a matrix by applying div_term elementwise to the positional matrix and taking
        # the sine and cosine of each odd and even element, respectively
        pe = torch.zeros_like(data.x)
        pe[:, 0::2] = torch.sin(position * div_term) ##even
        pe[:, 1::2] = torch.cos(position * div_term)  ##odd
        data.x = data.x+pe
        
        return data


class Normal_Sym_Image_Dataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, mode="train"):

        self.npath = img_dir  # os.path.join(img_dir, "noisy")
        self.nimg_name = sorted(os.listdir(self.npath))
        self.nimg_name = [
            i
            for i in self.nimg_name
            if i.split(".")[-1].lower() in ["jpeg", "jpg", "png", "bmp", "tif"]
        ]
        self.transform = transform
        self.target_transform = target_transform
        self.pth = "/home/jyh_temp1/Downloads/GraphLocalDenoising/eye_image/sym_normal/" + mode + "/graph/"
        ###"iter"+str(self.iter_num)+"compact"+str(self.compact)+
        if os.path.exists(self.pth) and len(os.listdir(self.pth)) > 1:
            pass
        else:
            os.makedirs(self.pth, exist_ok=True)
            self.preprocess_graph()
        super(Normal_Image_Dataset, self).__init__()

    def len(self):
        return len(self.nimg_name)

    def save(self, idx):
        ksize = 4
        stride = 4
        padding = 0
        tmp_train_name = self.nimg_name[idx]
        nimg_name = os.path.join(self.npath, tmp_train_name)
        img_x = Image.open(nimg_name).convert('L')  ############读取路径指向图片，非numpy类型
        ######empoly SLIC get superpxl
        origin_img = cv2.resize(np.array(img_x), (440, 440))
        gt_mask_image = np.zeros_like(origin_img)
        h = origin_img.shape[0]
        w = origin_img.shape[1]
        row_num = h / stride
        col_num = w / stride
        num_nodes = int(row_num * col_num)
        img = torch.from_numpy(origin_img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img.float()
        gt_mask = torch.from_numpy(gt_mask_image)
        gt_mask = gt_mask.unsqueeze(0)
        gt_mask = gt_mask.unsqueeze(0)
        gt_mask = gt_mask.float()
        unfold = torch.nn.Unfold(kernel_size=ksize, stride=stride, padding=padding)
        # print("img:",img,img.shape)
        patches = unfold(img)
        patches = (patches.squeeze(0)).squeeze(0)
        feat_mat = (patches.T)  #####num_nodes*size
        ##mask
        gt_patches = unfold(gt_mask)
        gt_patches = (gt_patches.squeeze(0)).squeeze(0)
        gt_mask_node = 1 - (gt_patches.T)
        ###adj construction using symetric prior across the center line(need position info)
        adj1 = local_adj(feat_mat, int(row_num), int(col_num))
        '''nonlocal adjacency'''
        start = time.time()
        adj2 = 0  # nonlocal_adj(feat_mat,int(row_num),int(col_num),thres=8000)
        end = time.time()
        print("adjacency completed, cost time:", end - start)
        adj = adj1 + adj2
        adj[adj > 1] = 1
        adj = sp.coo_matrix(adj)
        values = adj.data
        indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
        adj = torch.LongTensor(indices)  # PyG框架需要的coo形式
        ##instance edge is binary label that only connects superpixels within one nuclie
        data = Data(x=feat_mat, edge_index=adj, y=gt_mask_node, img=torch.from_numpy(origin_img),
                    img_y=torch.from_numpy(gt_mask_image))
        torch.save(data, self.pth + str(idx) + ".pth")
        print("saved for " + str(idx))

    def preprocess_graph(self):
        # self.mean,self.std = self.get_all_sample_mean()
        p_map(self.save, range(len(self.nimg_name)))  #

    def get(self, idx):
        data = torch.load(self.pth + str(idx) + ".pth")
        ##add postion
        data.feat_num = data.x.shape[1]
        position = torch.arange(0, data.x.shape[0]).unsqueeze(1)
        # create matrix of divisors (powers of 10000) with exponents ranging from 0 to d_model-1
        div_term = torch.exp(
            torch.arange(0, data.feat_num, 2) * -(math.log(10000.0) / data.feat_num))  ##data.feat_num must be even
        # create a matrix by applying div_term elementwise to the positional matrix and taking
        # the sine and cosine of each odd and even element, respectively
        pe = torch.zeros_like(data.x)
        pe[:, 0::2] = torch.sin(position * div_term)  ##even
        pe[:, 1::2] = torch.cos(position * div_term)  ##odd
        data.x = data.x + pe

        return data
