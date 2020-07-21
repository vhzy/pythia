import torch
import torch.nn as nn
import numpy as np
import argparse
import math
import torch.nn.functional as F

class GraphConvNet(nn.Module):
    def __init__(self, dim):
        super(GraphConvNet, self).__init__()
        self.dim = dim
        self.fc = nn.Linear(self.dim, self.dim)

    def forward(self, inputs, adj_mat): # inputs: bs x N x dim  adj_mat: bs x N x N
        y = torch.matmul(adj_mat, self.fc(inputs))
        return  y

class GatedGraphConvNet(nn.Module):
    def __init__(self, dim):
        super(GatedGraphConvNet, self).__init__()
        self.dim = dim
        self.fc_inp = nn.Linear(self.dim, self.dim)
        self.fc_u_x = nn.Linear(self.dim, self.dim)
        self.fc_u_y = nn.Linear(self.dim, self.dim)
        self.fc_r_y = nn.Linear(self.dim, self.dim)
        self.fc_r_x = nn.Linear(self.dim, self.dim)
        self.fc_t_y = nn.Linear(self.dim, self.dim)
        self.fc_t_x = nn.Linear(self.dim, self.dim)

    def forward(self, inputs, adj_mat): # inputs : bs x N x dim  adj_mat: bs x N x N
        y = torch.matmul(adj_mat, self.fc_inp(inputs)) # bs x N x N * bs x N x dim -> bs x N x dim
        u = torch.sigmoid(self.fc_u_y(y) + self.fc_u_x(inputs)) #
        r = torch.sigmoid(self.fc_r_y(y) + self.fc_r_x(inputs))
        x_tmp = torch.tanh(self.fc_t_y(y) + self.fc_t_x(r*inputs))
        out = (1 - u) * inputs + u * x_tmp
        return out

class MultiStepGGCN(nn.Module):
    def __init__(self, dim):
        super(MultiStepGGCN, self).__init__()
        self.gcn_layers = 1
        self.num_dim = dim
        self.ggcn = GatedGraphConvNet(dim)

    def forward(self, inputs, adj_mat):
        for i in range(self.gcn_layers):
            inputs = self.ggcn(inputs, adj_mat)
        return inputs

class BaseGraphAttNet(nn.Module):
    def __init__(self, args_params):
        super(BaseGraphAttNet, self).__init__()
        num_hid = args_params.num_hid
        self.fc = nn.Linear(num_hid, num_hid)
        self.leaky_relu = nn.LeakyReLU()
        self.q_fc = nn.Linear(num_hid, 1)
        self.k_fc = nn.Linear(num_hid, 1)
        self.dropout = nn.Dropout(args_params.dropout_rate)

    def forward(self, feats, adj_mat, residual=True):
        """
        :param feats: bs x N x dim
        :param adj_mat: bs x N x N
        :param residual: True
        :return: bs x N x num_dim
        """
        feat_proj = self.fc(feats)
        q_feats = self.q_fc(feat_proj)
        k_feats = self.k_fc(feat_proj)
        logits = q_feats + torch.transpose(k_feats, 2, 1)

        # option 1:
        masked_logits = logits + (1.0 - adj_mat) * -1e9
        masked_logits = self.leaky_relu(masked_logits)
        atten_value = F.softmax(masked_logits, dim=-1)

        atten_value = self.dropout(atten_value)
        output = torch.matmul(atten_value, feat_proj)
        if residual:
            output = output + feats
        return output

class GatedGraphAttNet(nn.Module):
    def __init__(self, config):
        super(GatedGraphAttNet, self).__init__()
        self.dim = config.num_hid
        self.gcn = BaseGraphAttNet(config)

        self.fc_u_x = nn.Linear(self.dim, self.dim)
        self.fc_u_y = nn.Linear(self.dim, self.dim)
        self.fc_r_y = nn.Linear(self.dim, self.dim)
        self.fc_r_x = nn.Linear(self.dim, self.dim)
        self.fc_t_y = nn.Linear(self.dim, self.dim)
        self.fc_t_x = nn.Linear(self.dim, self.dim)

    def forward(self, inputs, adj_mat):  # inputs : bs x N x dim  adj_mat: bs x N x N
        y = self.gcn(inputs, adj_mat, residual=False)  # bs x N x N * bs x N x dim -> bs x N x dim
        u = torch.sigmoid(self.fc_u_y(y) + self.fc_u_x(inputs))  #
        r = torch.sigmoid(self.fc_r_y(y) + self.fc_r_x(inputs))
        x_tmp = torch.tanh(self.fc_t_y(y) + self.fc_t_x(r * inputs))
        out = (1 - u) * inputs + u * x_tmp
        return out

class MultiHeadGraphAttNet(nn.Module):
    def __init__(self, dim, dropout_r=0.15):
        super(MultiHeadGraphAttNet, self).__init__()
        self.dim = dim
        self.head_num = 8

        self.fc = nn.Linear(dim, dim*self.head_num)
        self.leaky_relu = nn.LeakyReLU()
        self.q_fc = nn.Linear(self.dim, 1)
        self.k_fc = nn.Linear(self.dim, 1)
        self.dropout = nn.Dropout(dropout_r)

        self.final_proj = nn.Linear(dim*self.head_num, dim)

    def forward(self, feats, adj_mat, residual=True):
        bs = feats.size(0)
        N = feats.size(1)
        feat_proj = self.fc(feats) # bs x N x (n_head x dim)

        feat_proj = feat_proj.view(bs, N, self.head_num, self.dim).transpose(1, 2) # bs x n_head x N x dim
        q_feats = self.q_fc(feat_proj) # bs x n_head x N x 1
        k_feats = self.k_fc(feat_proj) # bs x n_head x N x 1
        logits = q_feats + torch.transpose(k_feats, 3, 2) # bs x n_head x N x N

        adj_mask = adj_mat.unsqueeze(1) # bs x 1 x N x N
        masked_logits = logits + (1.0 - adj_mask)* -1e9
        masked_logits = self.leaky_relu(masked_logits)
        atten_value = F.softmax(masked_logits, dim=-1) # bs x n_head x N x N

        atten_value = self.dropout(atten_value)
        output = torch.matmul(atten_value, feat_proj) # bs x n_head x N x dim
        output = torch.transpose(output, 1, 2).contiguous() # bs x N x n_head x dim
        output = output.view(bs, N, self.dim*self.head_num)
        output = self.final_proj(output)

        if residual:
            output = output + feats
        output = torch.sigmoid(output)
        return output

class AttFlat(nn.Module):
    def __init__(self, dim):
        super(AttFlat, self).__init__()
        self.mlp = nn.Linear(dim, 1)
        self.linear_merge = nn.Linear(dim, dim)

    def forward(self, x, x_mask=None): # x: bs x len x dim, x_mask: bs x len
        att = self.mlp(x) # bs x len x 1
        if x_mask is not None:
            #att = att.masked_fill(x_mask.unsqueeze(2), -1e9)
            x_mask = x_mask.unsqueeze(2)
            mask_flag = torch.ones_like(x_mask) * {-1e9}
            att = torch.where(x_mask>0, att, mask_flag)

        att = F.softmax(att, dim=1)

        x_atted = torch.sum(att * x, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

class QuestionConditionedGAT(nn.Module):
    def __init__(self, dim, dropout_r):
        super(QuestionConditionedGAT, self).__init__()
        self.dim = dim
        self.ques_flat = AttFlat(768)
        self.fc = nn.Linear(2*dim, dim)
        self.leaky_relu = nn.LeakyReLU()
        self.q_fc = nn.Linear(dim, 1)
        self.k_fc = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, ques_repr, feat_repr, adj_mat, residual): # bs x N x dim
        ques_flat_repr = self.ques_flat(ques_repr).unsqueeze(1) # bs x 1 x dim
        feat_len = feat_repr.size(1)
        ques_ext_repr = ques_flat_repr.repeat(1, feat_len, 1) # bs x N x dim
        feat_repr = torch.cat([feat_repr, ques_ext_repr], dim=-1) # bs x N x 2dim
        feat_proj = self.fc(feat_repr)
        q_feats = self.q_fc(feat_proj)
        k_feats = self.k_fc(feat_proj)
        logits = q_feats + torch.transpose(k_feats, 2, 1)

        # option 1:
        masked_logits = logits + (1.0 - adj_mat) * -1e9
        masked_logits = self.leaky_relu(masked_logits)
        atten_value = F.softmax(masked_logits, dim=-1)

        atten_value = self.dropout(atten_value)
        output = torch.matmul(atten_value, feat_proj)
        if residual:
            output = output + feat_repr
        return output

if __name__ == '__main__':
    ques_repr = torch.rand(8, 4, 16).cuda()
    feat = torch.rand(8, 5, 16).cuda()
    adj_mat = torch.rand(8, 5, 5)
    ones = torch.ones(8, 5, 5)
    zeros = torch.zeros(8, 5, 5)
    adj_mat = torch.where(adj_mat>0.5, ones, zeros).cuda()
    bgat = QuestionConditionedGAT(16, 0.15).cuda()
    out = bgat(ques_repr, feat, adj_mat)
    print(out.size())
