from pythia.modules.attention import AttFlat, SelfAttention, SoftAttention
import torch


if __name__ == '__main__':
    feat = torch.rand(1, 4, 12).cuda()
    print(feat,"\n")
    self_atten = SelfAttention(12).cuda()
    self_feat0=self_atten(feat)
    print("自注意力结果是:",self_feat0)
