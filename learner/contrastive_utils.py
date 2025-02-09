from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np


class PairConLoss(nn.Module):
    def __init__(self, temperature=0.05, m=0.4):
        super(PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        self.m = m
        print(f"\n Initializing PairConLoss \n")

    def forward(self, features_1, features_2):
        loss = self.MCL(features_1, features_2, self.m, self.temperature)
        return {"loss":loss}

    def mask_correlated(self, size):
        N = 2 * size
        mask = torch.ones((N, N)).to('cuda')
        mask = mask.fill_diagonal_(0)
        for i in range(size):
            mask[i, size + i] = 0
            mask[size + i, i] = 0
        mask = mask.bool()
        return mask

    def MCL(self, feature1, feature2, m, t):
        num_sample = feature1.size(0)
        z0 = torch.cat((feature1, feature2), dim=0)
        sim0 = torch.matmul(z0, z0.T) / t
        e_sim0 = torch.exp(sim0)

        sim_i_j = torch.diag(e_sim0, num_sample)
        sim_j_i = torch.diag(e_sim0, -num_sample)
        numerator1 = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * num_sample, 1)
        negative_samples = e_sim0[self.mask_correlated(num_sample)].reshape(2 * num_sample, -1)
        denominator1 = torch.sum(negative_samples, dim=1).reshape(2 * num_sample, 1)
        soft0 = numerator1 / denominator1
        log_soft1 = torch.log(soft0)
        cml_loss = - torch.mean(log_soft1)
        return cml_loss


class Attention_loss(nn.Module):
    def __init__(self, temperature=0.05):
        super(Attention_loss, self).__init__()
        self.t = temperature
        self.eps = 1e-08
        self.diagonal = 0

    def mask_correlated(self, size):
        N = 2 * size
        mask = torch.ones((N, N)).to('cuda')
        mask = mask.fill_diagonal_(0)
        for i in range(size):
            mask[i, size + i] = 0
            mask[size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, feature1, feature2, S_weight, pre_label):
        num_sample = pre_label.size(0)
        pre_label = pre_label.reshape(num_sample, 1)
        category_matrix = (pre_label == pre_label.T).float()
        category_matrix = category_matrix.repeat(2, 2)

        S = S_weight.repeat(2, 2)
        if self.diagonal == 0:
            S2 = S.fill_diagonal_(0)
        H = torch.cat((feature1, feature2), dim=0)
        sim2 = torch.matmul(H, H.T) / self.t
        e_sim2 = torch.exp(sim2)
        weight_e_sim2 = e_sim2 * (S2 + 1e-5) * category_matrix
        numerator2 = torch.sum(weight_e_sim2, dim=1).reshape(2 * num_sample, -1)
        negative_samples = e_sim2[self.mask_correlated(num_sample)].reshape(2 * num_sample, -1)
        denominator2 = torch.sum(negative_samples, dim=1).reshape(2 * num_sample, -1)
        soft2 = numerator2 / denominator2

        log_soft2 = torch.log(soft2)
        cml_loss = - torch.mean(log_soft2)

        return cml_loss