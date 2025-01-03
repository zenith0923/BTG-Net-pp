import torch
import torch.nn as nn

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

class Self_filtering(nn.Module):
    def __init__(self,resnet, in_c):
        super().__init__()
        if resnet:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.ELU(inplace=True),

            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.ELU(inplace=True),
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),

            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)

        return output
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


class SGNF(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        if resnet:
            self.num_channel = 640
            self.self_t = Self_filtering(resnet,in_c=160)
            self.self_s = Self_filtering(resnet,in_c=320)
            self.fiter_gs = nn.Sequential(
                nn.Conv2d(self.num_channel, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            )
            self.fiter_gt = nn.Sequential(
                nn.Conv2d(self.num_channel, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            )
            self.fc_s = Feature_compression(resnet, in_c=320)
            self.fc_t = Feature_compression(resnet, in_c=160)
        else:
            self.num_channel = 64
            self.self_t = Self_filtering(resnet,in_c=64)
            self.self_s = Self_filtering(resnet,in_c=64)
            self.fiter_gs = nn.Sequential(
                nn.Conv2d(self.num_channel, 30, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(30),
                nn.ReLU(inplace=True),
                nn.Conv2d(30, 1, kernel_size=1, stride=1, padding=0)
            )
            self.fiter_gt = nn.Sequential(
                nn.Conv2d(self.num_channel, 30, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(30),
                nn.ReLU(inplace=True),
                nn.Conv2d(30, 1, kernel_size=1, stride=1, padding=0)
            )
            self.fc_s = Feature_compression(resnet, in_c=64)
            self.fc_t = Feature_compression(resnet, in_c=64)

    def forward(self, F_t, F_s, F_l, way, shot):
        F_t = self.self_t(F_t)
        F_s = self.self_s(F_s)

        heat_map_s = nn.functional.interpolate(F_l, size=(F_s.shape[-1], F_s.shape[-1]), mode='bilinear', align_corners=False)
        fiter_s = nn.Sigmoid()(self.fiter_gs(heat_map_s))
        F_s = F_s * fiter_s

        heat_map_t = nn.functional.interpolate(F_l, size=(F_t.shape[-1], F_t.shape[-1]), mode='bilinear', align_corners=False)
        fiter_t = nn.Sigmoid()(self.fiter_gt(heat_map_t))
        F_t = F_t * fiter_t

        f_s = self.fc_s(F_s)
        f_t = self.fc_t(F_t)

        support_fs = f_s[:way * shot].view(way, shot, -1).mean(1)
        query_fs = f_s[way * shot:]

        support_ft = f_t[:way * shot].view(way, shot, -1).mean(1)
        query_ft = f_t[way * shot:]

        return support_fs, query_fs, support_ft, query_ft


