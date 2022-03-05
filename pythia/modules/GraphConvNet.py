import torch
import torch.nn as nn
import numpy as np
import argparse
import math
import torch.nn.functional as F
from pythia.modules.attention import AttFlat, SelfAttention, SoftAttention
import sys
sys.path.append("/home/ubuntu/hzy/pythia/pythia/modules")

class GraphConvNet(nn.Module):
    def __init__(self, dim):
        super(GraphConvNet, self).__init__()
        self.dim = dim
        self.fc = nn.Linear(self.dim, self.dim)

    def forward(self, inputs, adj_mat): # inputs: bs x N x dim  adj_mat: bs x N x N
        y = torch.matmul(adj_mat, self.fc(inputs))
        return y

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
    def __init__(self, num_hid, dropout_rate=0.15):
        super(BaseGraphAttNet, self).__init__()
        self.fc = nn.Linear(num_hid, num_hid)
        self.leaky_relu = nn.LeakyReLU()
        self.q_fc = nn.Linear(num_hid, 1)
        self.k_fc = nn.Linear(num_hid, 1)
        self.dropout = nn.Dropout(dropout_rate)

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

class MultiHeadGraphAttNet(nn.Module):
    def __init__(self, dim, n_heads, dropout_r=0.15):
        super(MultiHeadGraphAttNet, self).__init__()
        self.dropout = nn.Dropout(dropout_r)
        self.dim = dim
        self.attentions = nn.ModuleList([BaseGraphAttNet(dim, dropout_r) for _ in range(n_heads)])

        self.out_att = nn.Linear(dim*n_heads, dim)
        self.fc_u_x = nn.Linear(self.dim, self.dim)
        self.fc_u_y = nn.Linear(self.dim, self.dim)
        self.fc_r_y = nn.Linear(self.dim, self.dim)
        self.fc_r_x = nn.Linear(self.dim, self.dim)
        self.fc_t_y = nn.Linear(self.dim, self.dim)
        self.fc_t_x = nn.Linear(self.dim, self.dim)

    def forward(self, feats, adj_mat):
        comb_atts = torch.cat([att(feats, adj_mat) for att in self.attentions], dim=-1)
        logits = self.dropout(comb_atts)
        y = F.elu(self.out_att(logits))

        u = torch.sigmoid(self.fc_u_y(y) + self.fc_u_x(feats))  #
        r = torch.sigmoid(self.fc_r_y(y) + self.fc_r_x(feats))
        x_tmp = torch.tanh(self.fc_t_y(y) + self.fc_t_x(r * feats))
        out = (1 - u) * feats + u * x_tmp

        return out

class MHGATLayers(nn.Module):
    """
    Multi-Head Graph Attention layers
    """
    def __init__(self, dim, n_head, n_layers):
        super(MHGATLayers, self).__init__()
        self.layers = nn.ModuleList([MultiHeadGraphAttNet(dim, n_head) for _ in range(n_layers)])

    def forward(self, inp_feat, adj_mat):
        feats = inp_feat
        for layer in self.layers:
            feats = layer(feats, adj_mat)
        return feats

class QuesMHGATLayers(nn.Module):
    """
    question-conditioned Multi-Head Graph Attention Layers
    """
    def __init__(self, dim, n_head, n_layers):
        super(QuesMHGATLayers, self).__init__()
        self.ques_dim = dim
        self.fc = nn.Linear(dim+dim, dim)
        self.ques_flat = AttFlat(dim)
        self.mhgat_layers = MHGATLayers(dim, n_head, n_layers)

    def forward(self, ques_repr, feat_repr, adj_mat):

        ques_flat_repr = self.ques_flat(ques_repr).unsqueeze(1)
        feat_len = feat_repr.size(1)
        ques_ext_repr = ques_flat_repr.repeat(1, feat_len, 1)
        feat_repr_fc = torch.cat([feat_repr, ques_ext_repr], dim=-1)  # bs x N x 2dim
        feat_proj = self.fc(feat_repr_fc)
        logits = self.mhgat_layers(feat_proj, adj_mat)

        return logits

class QVConditionedGAT(nn.Module):
    def __init__(self, dim, dropout_r=0.15):
        super(QVConditionedGAT, self).__init__()
        self.dim = dim
        self.atten = SoftAttention(dim, dim, dim, dropout_r)
        self.fc = nn.Linear(2*dim, dim)
        self.leaky_relu = nn.LeakyReLU()
        self.q_fc = nn.Linear(dim, 1)
        self.k_fc = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, ques_repr, feat_repr, adj_mat, residual=True): # bs x N x dim
        ques_flat_repr = self.atten(ques_repr, feat_repr.sum(1), None).unsqueeze(1) # bs x 1 x dim
        feat_len = feat_repr.size(1)
        ques_ext_repr = ques_flat_repr.repeat(1, feat_len, 1) # bs x N x dim
        feat_repr_fc = torch.cat([feat_repr, ques_ext_repr], dim=-1) # bs x N x 2dim  公式8
        feat_proj = self.fc(feat_repr_fc) # 公式9
        q_feats = self.q_fc(feat_proj)
        k_feats = self.k_fc(feat_proj)
        logits = q_feats + torch.transpose(k_feats, 2, 1)
        #print("adj_mat:",adj_mat.size())
        #print("logits:",logits.size())

        # option 1:
        masked_logits = logits + (1.0 - adj_mat) * -1e9
        masked_logits = self.leaky_relu(masked_logits)    #公式10
        atten_value = F.softmax(masked_logits, dim=-1)    #公式11

        atten_value = self.dropout(atten_value)
        output = torch.matmul(atten_value, feat_proj)
        if residual:
            output = output + feat_repr
        return output


class QuestionConditionedGAT(nn.Module):
    def __init__(self, dim, dropout_r=0.15):
        super(QuestionConditionedGAT, self).__init__()
        self.dim = dim
        self.ques_flat = AttFlat(dim)
        self.fc = nn.Linear(2*dim, dim)
        self.leaky_relu = nn.LeakyReLU()
        self.q_fc = nn.Linear(dim, 1)
        self.k_fc = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, ques_repr, feat_repr, adj_mat, residual=True): # bs x N x dim
        ques_flat_repr = self.ques_flat(ques_repr).unsqueeze(1) # bs x 1 x dim
        feat_len = feat_repr.size(1)
        ques_ext_repr = ques_flat_repr.repeat(1, feat_len, 1) # bs x N x dim
        feat_repr_fc = torch.cat([feat_repr, ques_ext_repr], dim=-1) # bs x N x 2dim
        feat_proj = self.fc(feat_repr_fc)
        q_feats = self.q_fc(feat_proj)
        print("q_feats",q_feats.size())
        print(q_feats)
        k_feats = self.k_fc(feat_proj)
        print("k_feats",k_feats.size())
        print(k_feats)
        logits = q_feats + torch.transpose(k_feats, 2, 1)
        print("logits",logits.size())
        print(logits)
        # option 1:
        masked_logits = logits + (1.0 - adj_mat) * -1e9
        print("masked_logits",masked_logits.size())
        print(masked_logits)
        masked_logits = self.leaky_relu(masked_logits)
        atten_value = F.softmax(masked_logits, dim=-1)

        atten_value = self.dropout(atten_value)
        output = torch.matmul(atten_value, feat_proj)
        if residual:
            output = output + feat_repr
        return output

class QCGATLayers(nn.Module):
    def __init__(self, dim, num_gat_layers):
        super(QCGATLayers, self).__init__()
        self.num_gat_layers = num_gat_layers
        self.layers = nn.ModuleList([QuestionConditionedGAT(dim) for _ in range(self.num_gat_layers)])

    def forward(self, ques_repr, feat_repr, adj_mat):
        for layer in self.layers:
            feat_repr = layer(ques_repr, feat_repr, adj_mat)
        return feat_repr

class QVGATLayers(nn.Module):
    def __init__(self, dim, num_gat_layers):
        super(QVGATLayers, self).__init__()
        self.num_gat_layers = num_gat_layers
        self.layers = nn.ModuleList([QVConditionedGAT(dim) for _ in range(self.num_gat_layers)])

    def forward(self, ques_repr, feat_repr, adj_mat):
        for layer in self.layers:
            feat_repr = layer(ques_repr, feat_repr, adj_mat)
        return feat_repr

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
