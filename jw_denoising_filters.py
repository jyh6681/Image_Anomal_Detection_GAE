import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import math
import sys
def soft_thresholding(x, soft_eta, mode):
    """
    Perform row-wise soft thresholding.
    The row wise shrinkage is specific on E(k+1) updating
    The element wise shrinkage is specific on Z(k+1) updating

    :param x: one block of target matrix, shape[num_nodes, num_features]
    :param soft_eta: threshold scalar stores in a torch tensor
    :param mode: model types selection "row" or "element"
    :return: one block of matrix after shrinkage, shape[num_nodes, num_features]

    """
    assert mode in ('element', 'row'), 'shrinkage type is invalid (element or row)'
    if mode == 'row':
        row_norm = torch.linalg.norm(x, dim=1).unsqueeze(1)
        row_norm.clamp_(1e-12)
        row_thresh = (F.relu(row_norm - soft_eta)+soft_eta) / row_norm
        out = x * row_thresh
    else:
        out = F.relu(x - soft_eta) - F.relu(-x - soft_eta)

    return out

def coumpute_wtv(w,v):
    wtv = torch.zeros_like(v[0])
    for i in range(len(v)):
        wtv+=torch.sparse.mm(w[i], v[i])
    return wtv
def hard_thresholding(x, soft_eta, mode):
    """
    Perform row-wise hard thresholding.
    The row wise shrinkage is specific on E(k+1) updating
    The element wise shrinkage is specific on Z(k+1) updating

    :param x: one block of target matrix, shape[num_nodes, num_features]
    :param soft_eta: threshold scalar stores in a torch tensor
    :param mode: model types selection "row" or "element"
    :return: one block of matrix after shrinkage, shape[num_nodes, num_features]

    """
    assert mode in ('element', 'row'), 'shrinkage type is invalid (element or row)'
    tmp = torch.zeros_like(x)
    tmp[x - soft_eta>0] = 1
    tmp[-x-soft_eta>0] = 1
    # tmp[-x - soft_eta < 0]=0
    # tmp[x - soft_eta < 0] = 0
    #print("tmp:",torch.max(tmp))
    out = x*tmp
    return out


# Node denoising filter
class NodeDenoisingADMM(nn.Module):
    def __init__(self, num_nodes, num_features, r, J, nu, admm_iter, rho, gamma_0):
        super(NodeDenoisingADMM, self).__init__()
        self.r = r
        self.J = J
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.admm_iter = admm_iter
        self.rho = rho
        self.nu = [nu] * J
        for i in range(J):
            self.nu[i] = self.nu[i] / np.power(4.0, i)  # from (4.3) in Dong's paper
        self.nu = [0.0] + self.nu  # To include W_{0,J}
        self.gamma_max = 1e+6
        self.initial_gamma = gamma_0
        self.gamma = self.initial_gamma

    def forward(self, F, W_list, d, mask, init_Zk=None, init_Yk=None,lp=1,lq=2,boost=False,stop_thres=0.05,boost_value=4,thres_iter=15):
        """
        Parameters
        ----------
        F : Graph signal to be smoothed, shape [Num_node, Num_features].
        W_list : Framelet Base Operator, in list, each is a sparse matrix of size Num_node x Num_node.
        d : Vector of normalized graph node degrees in shape [Num_node, 1].
        init_Zk: Initialized list of (length: j * l) zero matrix in shape [Num_node, Num_feature].
        init_Yk: Initialized lists of (length: j*l) zero matrix in shape [Num_node, Num_feature].

        :returns:  Smoothed graph signal U

        """
        if init_Zk is None:
            Zk = []
            for j in range(self.r-1):
                for l in range(self.J):
                    Zk.append(torch.zeros(torch.Size([self.num_nodes, self.num_features])).to(F.device))
            Zk = [torch.zeros((self.num_nodes, self.num_features)).to(F.device)] + Zk
        else:
            Zk = init_Zk
        if init_Yk is None:
            Yk = []
            for j in range(self.r-1):
                for l in range(self.J):
                    Yk.append(torch.zeros(torch.Size([self.num_nodes, self.num_features])).to(F.device))
            Yk = [torch.zeros((self.num_nodes, self.num_features)).to(F.device)] + Yk
        else:
            Yk = init_Yk

        energy = 1000000000
        self.gamma = self.initial_gamma
        vk = [Yk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]
        Uk = F
        diff = 10000
        k = 1
        ak = boost_value
        v_til = vk
        energy_list = []
        diff_list = []
        #while diff>stop_thres or k<thres_iter:
        while k<thres_iter:
            if lp==1:
                Zk = [soft_thresholding(torch.sparse.mm(W_jl, Uk) + Yk_jl / self.gamma, (nu_jl / self.gamma)* d.unsqueeze(1), 'element')
                      for nu_jl, W_jl, Yk_jl in zip(self.nu, W_list, Yk)]
            if lp==0:
                Zk = [hard_thresholding(torch.sparse.mm(W_jl, Uk) + Yk_jl / self.gamma, (nu_jl / self.gamma) * d.unsqueeze(1), 'element')
                      for nu_jl, W_jl, Yk_jl in zip(self.nu, W_list, Yk)]
            if lq==2:
                if boost == 0:
                    v_til = [ Yk_jl - self.gamma * Zk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]
                if boost == 1:
                    #boosta = (k - 1) / (k + boost_value)
                    boosta= (1+math.sqrt(1+4*ak*ak))/2
                    v_til = [Yk_jl - self.gamma * Zk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]
                    v_til = [item + ((ak-1)/boosta)*(item-item0) for item0,item in zip(v_til0,v_til)]
                    v_til0 = v_til
                U_init = Uk
                WTV = coumpute_wtv(W_list,v_til)
                Uk = (d.unsqueeze(1) *F*mask*mask - WTV)/(d.unsqueeze(1) *mask*mask+self.gamma)
            if lq==1:
                if boost == 0:
                    v_til = [Yk_jl - self.gamma * Zk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]####
                if boost == 1:
                    boosta= (1+math.sqrt(1+4*ak*ak))/2
                    v_til = [Yk_jl - self.gamma * Zk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]
                    v_til = [item + ((ak-1)/boosta)*(item-item0) for item0,item in zip(v_til0,v_til)]
                    v_til0 = v_til
                WTV = coumpute_wtv(W_list, v_til)
                #WTV = torch.sparse.mm(torch.cat(W_list, dim=1), torch.cat(v_til, dim=0))##############computation cost
                Yk = soft_thresholding(-F-WTV/self.gamma, (1/2*self.gamma)*d.unsqueeze(1),'element')
                Yk = mask*Yk + (1-mask)*(-F-WTV/self.gamma)
                U_init = Uk
                Uk = Yk + F
            if boost == 0:
                Yk = [Yk_jl + self.gamma * (torch.sparse.mm(W_jl, Uk) - Zk_jl) for W_jl, Yk_jl, Zk_jl in
                      zip(W_list, Yk, Zk)]
            if boost == 1:
                boosta = (1 + math.sqrt(1 + 4 * ak * ak)) / 2
                Y0 = Yk
                Yk = [Yk_jl + self.gamma * (torch.sparse.mm(W_jl, Uk) - Zk_jl) for W_jl, Yk_jl, Zk_jl in
                      zip(W_list, Yk, Zk)]
                Yk = [item1 + ((ak-1)/boosta)*(item1-item0) for item1, item0 in zip(Yk,Y0)]
                ak = boosta

            if lp==1:
                energy_1 = [nu_jl * torch.sum(d.unsqueeze(1)*torch.abs(torch.sparse.mm(W_jl ,Uk))) for
                          nu_jl, W_jl in zip(self.nu, W_list)]
            if lp==0:
                energy_1 = [torch.nonzero(nu_jl*d.unsqueeze(1) * torch.sparse.mm(W_jl, Uk)).shape[0] for
                            nu_jl, W_jl in zip(self.nu, W_list)]
            if lq==2:
                energy2 = 0.5 * torch.sum(d.unsqueeze(1)*torch.pow(mask*(Uk - F), 2))
            if lq==1:
                energy2 = 0.5 * torch.sum(d.unsqueeze(1) * torch.abs(mask * (Uk - F)))
            energy = sum(energy_1) + energy2
            diff = torch.sum(torch.pow(Uk - U_init, 2)).item()
            k += 1
            if k>thres_iter:
                break
        return Uk,energy_list,diff_list



