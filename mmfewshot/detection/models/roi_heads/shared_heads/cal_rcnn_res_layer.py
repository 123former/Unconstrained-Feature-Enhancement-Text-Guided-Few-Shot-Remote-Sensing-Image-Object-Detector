# Copyright (c) OpenMMLab. All rights reserved.
import pdb

import torch.nn as nn
from mmdet.models.builder import SHARED_HEADS
from mmdet.models.roi_heads import ResLayer
from torch import Tensor
import torch
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, k_dim, v_dim, num_heads=1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        # 定义线性投影层，用于将输入变换到多头注意力空间
        self.proj_q = nn.Linear(in_dim, k_dim * num_heads, bias=False)
        self.proj_k = nn.Linear(in_dim, k_dim * num_heads, bias=False)
        self.proj_v = nn.Linear(in_dim, v_dim * num_heads, bias=False)
        # 定义多头注意力的线性输出层
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, in_dim = x.size()
        # 对输入进行线性投影, 将每个头的查询、键、值进行切分和拼接
        q = self.proj_q(x).view(batch_size, seq_len, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k = self.proj_k(x).view(batch_size, seq_len, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v = self.proj_v(x).view(batch_size, seq_len, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        # 计算注意力权重和输出结果
        attn = torch.matmul(q, k) / self.k_dim ** 0.5  # 注意力得分

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)  # 注意力权重参数
        output = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)  # 输出结果
        # 对多头注意力输出进行线性变换和输出
        output = self.proj_o(output)

        return output


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads=1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x1, x2, mask=None):
        len_s1, _ = x1.size()
        len_s2, _ = x2.size()

        q1 = self.proj_q1(x1).view(1, len_s1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(1, len_s2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(1, len_s2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(len_s1, -1)
        output = self.proj_o(output)
        output = self.act(output)

        return output


class SelAttention(nn.Module):
    # 用来实现mask-attention layer
    def __init__(self, in_dim, k_dim, v_dim, out_dim=None, num_heads=1, Sigmoid_flag=False):
        super(SelAttention, self).__init__()
        if out_dim is None:
            out_dim = in_dim

        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim, k_dim * num_heads, bias=False)
        self.proj_k1 = nn.Linear(in_dim, k_dim * num_heads, bias=False)
        self.proj_v1 = nn.Linear(in_dim, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, out_dim)

        if Sigmoid_flag:
            self.act = nn.Sigmoid()
        else:
            self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x1, mask=None):
        len_s, _ = x1.size()
        q1 = self.proj_q1(x1).view(1, len_s, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k1 = self.proj_k1(x1).view(1, len_s, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v1 = self.proj_v1(x1).view(1, len_s, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k1) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v1).permute(0, 2, 1, 3).contiguous().view(len_s, -1)
        output = self.proj_o(output)
        output = self.act(output)
        return output


@SHARED_HEADS.register_module()
class CalRCNNResLayer(ResLayer):
    """Shared resLayer for metarcnn and fsdetview.

    It provides different forward logics for query and support images.
    """

    def __init__(self, inchannel=2048, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()
        self.c2c = SelAttention(inchannel, 512, inchannel)
        self.c2b = SelAttention(inchannel, 512, inchannel * 3, out_dim=1, Sigmoid_flag=True)
        self.p2p = SelAttention(inchannel, 512, inchannel)
        self.c2p = CrossAttention(inchannel, inchannel, 512, inchannel)

        self.pro = nn.Linear(inchannel, inchannel * 3, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for query images.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).

        Returns:
            Tensor: Shape of (N, C).
        """
        res_layer = getattr(self, f'layer{self.stage + 1}')
        out = res_layer(x)
        out = out.mean(3).mean(2)
        return out

    def forward_support(self, x: Tensor) -> Tensor:
        """Forward function for support images.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).

        Returns:
            Tensor: Shape of (N, C).
        """
        x = self.max_pool(x)
        res_layer = getattr(self, f'layer{self.stage + 1}')
        out = res_layer(x)
        out = self.sigmoid(out)
        out = out.mean(3).mean(2)
        return out

    def aug_feat(self, support_feats, query_feats):
        support_feats_ = self.c2c(support_feats)
        query_feats_temp = self.p2p(query_feats)
        query_feats_ = self.c2p(query_feats, support_feats_)
        query_feats_ = torch.cat([query_feats, query_feats_temp, query_feats_], 1)
        H = self.c2b(support_feats_)
        background = torch.matmul(H.transpose(0, 1), support_feats_).contiguous()
        support_feats_ = torch.cat([support_feats_, background], 0)

        support_feats_ = self.pro(support_feats_)
        support_feats_ = self.act(support_feats_)

        return support_feats_, query_feats_
