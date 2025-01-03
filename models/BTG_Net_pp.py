import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.backbones import Conv_4, ResNet
import math

from models.module.GKPM import GKPM
from models.module.SGNF import SGNF
from utils.l2_norm import l2_norm

class BTG_Net_pp(nn.Module):

    def __init__(self, args=None):

        super().__init__()
        self.args = args
        self.shots = [self.args.train_shot, self.args.train_query_shot]
        self.way = self.args.train_way
        self.resnet = self.args.resnet
        if self.resnet:
            self.num_channel = 640
            self.feature_extractor = ResNet.resnet12(drop=True)
        else:
            self.num_channel = 64
            self.feature_extractor = Conv_4.BackBone(self.num_channel)

        self.sgnf = SGNF(self.resnet)
        self.gkpm = GKPM(self.resnet)
        self.scale_t = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.scale_s = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.scale_l = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def get_cosine_dist(self, inp, way, shot):

        F_t, F_s, F_L = self.feature_extractor(inp)

        support_fs, query_fs, support_ft, query_ft = self.sgnf(F_t, F_s, F_L, way, shot)

        support_fl, query_fl, l_con = self.gkpm(F_L, way, shot)


        cos_fl = F.linear(l2_norm(query_fl), l2_norm(support_fl))
        cos_fs = F.linear(l2_norm(query_fs), l2_norm(support_fs))
        cos_ft = F.linear(l2_norm(query_ft), l2_norm(support_ft))


        return cos_fl, cos_fs, cos_ft, l_con


    def meta_test(self, inp, way, shot):

        cos_fl, cos_fs, cos_ft, _ = self.get_cosine_dist(inp=inp, way=way, shot=shot)
        scores = cos_fl + cos_fs + cos_ft


        _, max_index = torch.max(scores, 1)
        return max_index

    def forward(self, inp):

        cos_fl, cos_fs, cos_ft, l_con = self.get_cosine_dist(inp=inp, way=self.way, shot=self.shots[0])
        cos_fl = cos_fl * self.scale_l
        cos_fs = cos_fs * self.scale_s
        cos_ft = cos_ft * self.scale_t

        return cos_fl, cos_fs, cos_ft, l_con
