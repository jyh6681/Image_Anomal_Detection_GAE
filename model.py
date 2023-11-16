import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv,VGAE#,GATConv
class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GATConv(nfeat, nhid)#GraphConvolution(nfeat, nhid) ###too large adj
        self.bn1 = nn.BatchNorm1d(nhid)
        self.gc2 = GATConv(nhid, nhid)##GraphConvolution(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.bn2(x)

        return x

class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()

        self.gc1 = GATConv(nhid, nhid)##GraphConvolution(nhid, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.gc2 = GATConv(nhid, nfeat)#GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x

class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(Dominant, self).__init__()
        
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        # self.struct_decoder = Structure_Decoder(hidden_size, dropout)
    
    def forward(self, data):
        # print("data:::",data)
        x = data.x
        adj = data.edge_index
        # x = [tmp_data.x for tmp_data in data]
        # x = torch.cat(x).cuda()
        # edge_index = [tmp_data.edge_index for tmp_data in data]
        # adj = torch.cat(edge_index).cuda()
        # encode
        x = self.shared_encoder(x, adj)
        # decode feature matrix
        x_hat = self.attr_decoder(x, adj)
        # decode adjacency matrix
        #struct_reconstructed = self.struct_decoder(x, adj)
        # return reconstructed matrices
        diff = torch.abs(data.x-x_hat)#torch.abs(data.x[:,1:]-x_hat[:,1:])#F.tanh()
        # origin_diff = torch.abs(data.x-x_hat)
        # patch_num = 110*110
        # diff = diff.reshape(int(data.x.shape[0]/patch_num), patch_num*16)
        # # 对每一行进行最大值池化
        # max_values, _ = torch.max(diff, dim=1)
        # max_values = max_values.repeat(patch_num,16)
        # max_values = max_values.flatten()
        # # 将结果重塑为原始矩阵的形状
        # max_values = max_values.reshape(data.x.shape[0], 16)
        # # print("shape:::",data.x.shape,origin_diff.shape,diff.shape,max_values.shape)
        # mask = F.tanh(origin_diff/max_values)
        # new_mask = mask.clone()
        # new_mask[mask>0.6]=1
        # new_mask[mask<0.6]=0
        return diff,x_hat