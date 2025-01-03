import torch
import torch.nn as nn
from einops import rearrange



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False, max=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        self.max = nn.MaxPool2d(2) if max else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.max is not None:
            x = self.max(x)
        return x

class Feature_compression(nn.Module):
    def __init__(self, resnet, in_c):
        super().__init__()
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.alpha = 0.3
        if resnet:
            self.feature_size = 640
            self.conv_block = nn.Sequential(
                BasicConv(in_c, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True)
            )
            self.mlp = nn.Sequential(
                nn.BatchNorm1d(self.feature_size),
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )
        else:
            self.feature_size = 64
            self.conv_block = nn.Sequential(
                BasicConv(in_c, self.feature_size, kernel_size=3, stride=1, padding=1, relu=True, bias=True)
            )
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.alpha * x + (1 - self.alpha) * self.mlp(x)
        return x
class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, head_dim_ratio=1., qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = round(dim // num_heads * head_dim_ratio)
        self.head_dim = head_dim
        qk_scale_factor = qk_scale if qk_scale is not None else -0.25
        self.scale = head_dim ** qk_scale_factor
        self.qkv = nn.Linear(dim, head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_s = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.proj_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, s, TG_prompt):
        B_s = s.shape[0]

        t = s + TG_prompt
        c = torch.cat([t, query], dim=0)
        B, C = c.shape

        x = self.qkv(c)

        qkv = rearrange(x, 'b (x y) -> x b y', x=3, y=self.head_dim, b=B)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = ((q * self.scale) @ (k.transpose(-2, -1) * self.scale))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_cq = attn[:B_s, B_s:]
        attn_qc = attn[B_s:, :B_s]

        x_s = torch.matmul(attn_cq, query)
        x_q = torch.matmul(attn_qc, s)

        x_s = self.proj_s(x_s)
        x_s = self.proj_drop(x_s)
        x_q = self.proj_q(x_q)
        x_q = self.proj_drop(x_q)

        return x_s, x_q

class GKPM(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.alpha = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        if resnet:
            self.num_channel = 640
            self.TG_prompt = nn.Parameter(torch.randn((1, self.num_channel)))
            self.attn = Attention(self.num_channel, num_heads=1)
            self.fc_l = Feature_compression(resnet, in_c=640)
        else:
            self.num_channel = 64
            self.TG_prompt = nn.Parameter(torch.randn((1, self.num_channel)))
            self.attn = Attention(self.num_channel, num_heads=1)
            self.fc_l = Feature_compression(resnet, in_c=64)

    def forward(self, F_l, way, shot):
        f_l = self.fc_l(F_l)
        support_fl = f_l[:way * shot].view(way, shot, -1).mean(1)
        query_fl = f_l[way * shot:]
        x_s, x_q = self.attn(query_fl, support_fl, self.TG_prompt)

        support_fl = support_fl + self.beta * x_s
        query_fl = query_fl + self.alpha * x_q

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        score = cos(support_fl, x_s)
        l_con = 1 - torch.mean(score)

        return support_fl, query_fl, l_con
