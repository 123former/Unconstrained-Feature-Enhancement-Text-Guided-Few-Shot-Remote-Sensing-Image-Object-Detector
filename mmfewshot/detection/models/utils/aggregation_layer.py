# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.utils import ConfigDict
from mmdet.models.builder import MODELS
from torch import Tensor
import pdb

# AGGREGATORS are used for aggregate features from different data
# pipelines in meta-learning methods, such as attention rpn.
AGGREGATORS = MODELS


def build_aggregator(cfg: ConfigDict) -> nn.Module:
    """Build aggregator."""
    return AGGREGATORS.build(cfg)


@AGGREGATORS.register_module()
class AggregationLayer(BaseModule):
    """Aggregate query and support features with single or multiple aggregator.
    Each aggregator return aggregated results in different way.

    Args:
        aggregator_cfgs (list[ConfigDict]): List of fusion function.
        init_cfg (ConfigDict | None): Initialization config dict. Default: None
    """

    def __init__(self,
                 aggregator_cfgs: List[ConfigDict],
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.aggregator_list = nn.ModuleList()
        self.num_aggregators = len(aggregator_cfgs)
        aggregator_cfgs_ = copy.deepcopy(aggregator_cfgs)
        for cfg in aggregator_cfgs_:
            self.aggregator_list.append(build_aggregator(cfg))

    def forward(self, query_feat: Tensor,
                support_feat: Tensor) -> List[Tensor]:
        """Return aggregated features of query and support through single or
        multiple aggregators.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with
                shape (N, C, H, W).

        Returns:
            list[Tensor]: List of aggregated features.
        """
        out = []
        for i in range(self.num_aggregators):
            out.append(self.aggregator_list[i](query_feat, support_feat))
        return out


@AGGREGATORS.register_module()
class DepthWiseCorrelationAggregator(BaseModule):
    """Depth-wise correlation aggregator.

    Args:
        in_channels (int): Number of input features channels.
        out_channels (int): Number of output features channels.
            Default: None.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert in_channels is not None, \
            "DepthWiseCorrelationAggregator require config of 'in_channels'."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_fc = with_fc
        if with_fc:
            assert out_channels is not None, 'out_channels is expected.'
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.LayerNorm([out_channels])
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (1, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        assert query_feat.size(1) == support_feat.size(1), \
            'mismatch channel number between query and support features.'

        feat = F.conv2d(query_feat, support_feat.permute(1, 0, 2, 3),groups=self.in_channels)
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            feat = self.fc(feat.squeeze(3).squeeze(2))
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat


@AGGREGATORS.register_module()
class DifferenceAggregator(BaseModule):
    """Difference aggregator.

    Args:
        in_channels (int): Number of input features channels. Default: None.
        out_channels (int): Number of output features channels. Default: None.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.with_fc = with_fc
        self.in_channels = in_channels
        self.out_channels = out_channels
        if with_fc:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.LayerNorm([out_channels])
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (N, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        assert query_feat.size(1) == support_feat.size(1), \
            'mismatch channel number between query and support features.'
        feat = query_feat - support_feat
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            feat = self.fc(feat.squeeze(3).squeeze(2))
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat


@AGGREGATORS.register_module()
class DotProductAggregator(BaseModule):
    """Dot product aggregator.

    Args:
        in_channels (int): Number of input features channels. Default: None.
        out_channels (int): Number of output features channels. Default: None.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.with_fc = with_fc
        self.in_channels = in_channels
        self.out_channels = out_channels
        if with_fc:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (N, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """

        assert query_feat.size()[1:] == support_feat.size()[1:], \
            'mismatch channel number between query and support features.'
        feat = query_feat.mul(support_feat)
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            feat = self.fc(feat.squeeze(3).squeeze(2))
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat


@AGGREGATORS.register_module()
class MultiRelationCol(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(MultiRelationCol, self).__init__()
        self.key_t = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.value_t = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.key_q = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.value_q = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

        # self.bnn0 = nn.BatchNorm2d(256)
        # self.bnn1 = nn.BatchNorm2d(256)

    def forward(self, query_feats, support_feats):
        num_instance = query_feats.shape[0]
        num_cls = support_feats.shape[0]

        key_t = self.key_t(support_feats)
        val_t = self.value_t(support_feats)

        key_q = self.key_q(query_feats)
        val_q = self.value_q(query_feats)

        kq = key_q.permute(1, 0, 2, 3).view(256, num_instance, -1)
        kt = key_t.permute(1, 0, 2, 3)

        p = torch.matmul(kq.permute(2, 1, 0), kt.view(256, num_cls, -1).permute(2, 0, 1))
        p = F.softmax(p, dim=1)

        val_t_out = torch.matmul(val_t.view(num_cls, 1024, -1).permute(2, 1, 0), p.permute(0, 2, 1)).permute(2, 1, 0).view(
            num_instance, 1024, 7, 7)

        final = torch.cat((val_t_out, val_q), dim=1)#.mean(3).mean(2)
        return final


@AGGREGATORS.register_module()
class MultiCaseRelationCol(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(MultiCaseRelationCol, self).__init__()

        self.key_t = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.value_t = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.key_q = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.value_q = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.case_key_multi_t = nn.Conv2d(indim * 3, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.case_value_multi_t = nn.Conv2d(indim * 3, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.case_key_un_t = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.case_value_un_t = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.case_key_multi_q = nn.Conv2d(indim * 3, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.case_value_multi_q = nn.Conv2d(indim * 3, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.case_key_un_q = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.case_value_un_q = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, query_feats, query_cases_feats, support_feats, support_cases_feats):
        case_key_multi_t = self.case_key_multi_t(support_cases_feats)
        case_value_multi_t = self.case_value_multi_t(support_cases_feats)

        case_key_un_t = self.case_key_un_t(support_feats)
        case_value_un_t = self.case_value_un_t(support_feats)

        case_p_t = torch.matmul(case_key_un_t, case_key_multi_t)
        case_p_t = F.softmax(case_p_t, dim=1)

        case_value_multi_out_t = torch.matmul(case_value_multi_t, case_p_t)

        support_final_aug = torch.cat((case_value_multi_out_t, case_value_un_t), dim=0)

        case_key_multi_q = self.case_key_multi_q(query_cases_feats)
        case_value_multi_q = self.case_value_multi_q(query_cases_feats)

        case_key_un_q = self.case_key_un_q(query_feats)
        case_value_un_q = self.case_value_un_q(query_feats)

        case_p_q = torch.matmul(case_key_un_q, case_key_multi_q)
        case_p_q = F.softmax(case_p_q, dim=1)

        case_value_multi_out_q = torch.matmul(case_value_multi_q, case_p_q)

        query_final_aug = torch.cat((case_value_multi_out_q, case_value_un_q), dim=0)

        key_t = self.key_t(support_final_aug)
        val_t = self.value_t(support_final_aug)

        key_q = self.key_q(query_final_aug)
        val_q = self.value_q(query_final_aug)

        p = torch.matmul(key_q, key_t)
        p = F.softmax(p, dim=1)

        val_t_out = torch.matmul(val_t, p)
        final = torch.cat((val_t_out, val_q), dim=0)
        return final
