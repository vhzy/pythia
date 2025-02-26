# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_bert import BertAttention, BertConfig, BertSelfAttention
from pythia.modules.layers import GatedTanh, ModalCombineLayer, TransformLayer
from torch.nn.utils.weight_norm import weight_norm
from pythia.modules.utils import MLP

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

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        cfg = BertConfig(hidden_size=dim, num_hidden_layers=1)
        self.atten = BertSelfAttention(cfg)

    def forward(self, x, x_mask=None):
        out = self.atten(x, x_mask)
        #out_avg = out[0].mean(1)
        out_avg = out[0]
        return out_avg

class AttentionLayer(nn.Module):
    def __init__(self, image_dim, question_dim, **kwargs):
        super(AttentionLayer, self).__init__()

        combine_type = kwargs["modal_combine"]["type"]
        combine_params = kwargs["modal_combine"]["params"]
        modal_combine_layer = ModalCombineLayer(
            combine_type, image_dim, question_dim, **combine_params
        )

        transform_type = kwargs["transform"]["type"]
        transform_params = kwargs["transform"]["params"]
        transform_layer = TransformLayer(
            transform_type, modal_combine_layer.out_dim, **transform_params
        )

        normalization = kwargs["normalization"]

        self.module = TopDownAttention(
            modal_combine_layer, transform_layer, normalization
        )

        if getattr(self.module, "out_dim"):
            self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class SoftAttention(nn.Module):
    def __init__(self, inp_dim1, inp_dim2, inter_hid, dropout_p):
        super(SoftAttention, self).__init__()
        self.inp1_proj = MLP(inp_dim1, inter_hid, inter_hid, dropout_r=dropout_p)
        self.inp2_proj = MLP(inp_dim2, inter_hid, inter_hid, dropout_r=dropout_p)
        self.dropout_l = nn.Dropout(dropout_p)
        self.linear = weight_norm(nn.Linear(inter_hid, 1), dim=None)

    def forward(self, inp1, inp2, inp1_mask=None):
        # inp1: bs x len x dim1, mask: bs x len
        # inp2: bs x dim2
        s_len = inp1.size(1)
        inp1_proj = self.inp1_proj(inp1) # bs x len x dim
        inp2_proj = self.inp2_proj(inp2) # bs x dim
        inp2_proj = inp2_proj.unsqueeze(1).repeat(1, s_len, 1)
        joint_repr = inp1_proj * inp2_proj
        joint_repr = self.dropout_l(joint_repr)
        logits = self.linear(joint_repr)
        if inp1_mask is not None:
            logits = logits + (1.0 - inp1_mask.unsqueeze(2)) * -1e9
        weights = nn.functional.softmax(logits, 1)
        weighted_inp1 = (inp1 * weights + inp1).sum(1)
        return weighted_inp1


class ConcatenationAttention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size):
        super(ConcatenationAttention, self).__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.fa = GatedTanh(image_feat_dim + txt_rnn_embeding_dim, hidden_size)
        self.lc = nn.Linear(hidden_size, 1)

    def forward(self, image_feat, question_embedding):
        _, num_location, _ = image_feat.shape
        question_embedding_expand = torch.unsqueeze(question_embedding, 1).expand(
            -1, num_location, -1
        )
        concat_feature = torch.cat((image_feat, question_embedding_expand), dim=2)
        raw_attention = self.lc(self.fa(concat_feature))
        # softmax across locations
        attention_weights = nn.functional.softmax(raw_attention, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class ProjectAttention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout=0.2):
        super(ProjectAttention, self).__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.fa_image = GatedTanh(image_feat_dim, hidden_size)
        self.fa_txt = GatedTanh(txt_rnn_embeding_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lc = nn.Linear(hidden_size, 1)

    def compute_raw_att(self, image_feat, question_embedding):
        num_location = image_feat.shape[1]
        image_fa = self.fa_image(image_feat)
        question_fa = self.fa_txt(question_embedding)
        question_fa_expand = torch.unsqueeze(question_fa, 1).expand(
            -1, num_location, -1
        )
        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)
        raw_attention = self.lc(joint_feature)
        return raw_attention

    def forward(self, image_feat, question_embedding):
        raw_attention = self.compute_raw_att(image_feat, question_embedding)
        # softmax across locations
        attention_weights = nn.functional.softmax(raw_attention, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class DoubleProjectAttention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout=0.2):
        super(DoubleProjectAttention, self).__init__()
        self.att1 = ProjectAttention(
            image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout
        )
        self.att2 = ProjectAttention(
            image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout
        )
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim

    def forward(self, image_feat, question_embedding):
        att1 = self.att1.compute_raw_att(image_feat, question_embedding)
        att2 = self.att2.compute_raw_att(image_feat, question_embedding)
        raw_attn_weights = att1 + att2
        # softmax across locations
        attention_weights = nn.functional.softmax(raw_attn_weights, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class TopDownAttention(nn.Module):
    EPS = 1.0e-08

    def __init__(self, combination_layer, transform_module, normalization):
        super(TopDownAttention, self).__init__()
        self.combination_layer = combination_layer
        self.normalization = normalization
        self.transform = transform_module
        self.out_dim = self.transform.out_dim

    @staticmethod
    def _mask_attentions(attention, image_locs):
        batch_size, num_loc, n_att = attention.size()
        tmp1 = attention.new_zeros(num_loc)
        tmp1[:num_loc] = torch.arange(0, num_loc, dtype=attention.dtype).unsqueeze(
            dim=0
        )

        tmp1 = tmp1.expand(batch_size, num_loc)
        tmp2 = image_locs.type(tmp1.type())
        tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
        mask = torch.ge(tmp1, tmp2)
        mask = mask.unsqueeze(dim=2).expand_as(attention)
        attention = attention.masked_fill(mask, 0)
        return attention

    def forward(self, image_feat, question_embedding, image_locs=None):
        # N x K x joint_dim
        joint_feature = self.combination_layer(image_feat, question_embedding)
        # N x K x n_att
        raw_attn = self.transform(joint_feature)

        if self.normalization.lower() == "softmax":
            attention = nn.functional.softmax(raw_attn, dim=1)
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)
                masked_attention_sum = torch.sum(masked_attention, dim=1, keepdim=True)
                masked_attention_sum += masked_attention_sum.eq(0).float() + self.EPS
                masked_attention = masked_attention / masked_attention_sum
            else:
                masked_attention = attention

        elif self.normalization.lower() == "sigmoid":
            attention = torch.sigmoid(raw_attn)
            masked_attention = attention
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)

        return masked_attention
