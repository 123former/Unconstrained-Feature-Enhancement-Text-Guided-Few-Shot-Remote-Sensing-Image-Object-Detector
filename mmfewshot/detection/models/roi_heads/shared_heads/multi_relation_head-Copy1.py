# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads import BBoxHead
from torch import Tensor
from mmdet.core import BboxOverlaps2D
import pdb

class MultiRelationHead(nn.Module):
    """BBox head for `Attention RPN <https://arxiv.org/abs/1908.01998>`_.

    Args:
        patch_relation (bool): Whether use patch_relation head for
            classification. Following the official implementation,
            `patch_relation` always be True, because only patch relation
            head contain regression head. Default: True.
        local_correlation (bool): Whether use local_correlation head for
            classification. Default: True.
        global_relation (bool): Whether use global_relation head for
            classification. Default: True.
    """

    def __init__(self,
                 patch_relation = True,
                 local_correlation = True,
                 global_relation = True,
                 in_channels = 2048):
        super(MultiRelationHead, self).__init__()
        # following the official implementation patch relation must be True,
        # because only patch relation head contain regression head
        self.patch_relation = True
        self.local_correlation = local_correlation
        self.global_relation = global_relation
        self.iou_threshold = 0.7
        self.in_channels = in_channels
        if self.patch_relation:
            self.patch_relation_branch = nn.Sequential(
                nn.Conv2d(
                    self.in_channels * 2,
                    int(self.in_channels / 4),
                    1,
                    padding=0,
                    bias=False),
                nn.ReLU(inplace=True),
                # 7x7 -> 5x5
                nn.AvgPool2d(kernel_size=3, stride=1),
                # 5x5 -> 3x3
                nn.Conv2d(
                    int(self.in_channels / 4),
                    int(self.in_channels / 4),
                    3,
                    padding=0,
                    bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    int(self.in_channels / 4),
                    self.in_channels,
                    1,
                    padding=0,
                    bias=False),
                nn.ReLU(inplace=True),
                # 3x3 -> 1x1
                nn.AvgPool2d(kernel_size=3, stride=1))
            self.patch_relation_fc_cls = nn.Linear(self.in_channels, 2)

        if self.local_correlation:
            self.local_correlation_branch = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.in_channels,
                    1,
                    padding=0,
                    bias=False))
            self.local_correlation_fc_cls = nn.Linear(self.in_channels, 2)

        if self.global_relation:
            self.global_relation_avgpool = nn.AvgPool2d(7)
            self.global_relation_branch = nn.Sequential(
                nn.Linear(self.in_channels * 2, self.in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.in_channels, self.in_channels),
                nn.ReLU(inplace=True))
            self.global_relation_fc_cls = nn.Linear(self.in_channels, 2)

    def forward(self, query_feat: Tensor,
                support_feat: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            query_feat (Tensor): Shape of (num_proposals, C, H, W).
            support_feat (Tensor): Shape of (1, C, H, W).

        Returns:
            tuple:
                cls_score (Tensor): Cls scores, has shape
                    (num_proposals, num_classes).
                bbox_pred (Tensor): Box energies / deltas, has shape
                    (num_proposals, 4).
        """

        # global_relation
        if self.global_relation:
            global_query_feat = self.global_relation_avgpool(
                query_feat).squeeze(3).squeeze(2)
            global_support_feat = self.global_relation_avgpool(
                support_feat).squeeze(3).squeeze(2).expand_as(
                    global_query_feat)
            global_feat = \
                torch.cat((global_query_feat, global_support_feat), 1)
            global_feat = self.global_relation_branch(global_feat)
            global_relation_cls_score = \
                self.global_relation_fc_cls(global_feat)

        # local_correlation
        if self.local_correlation:
            local_query_feat = self.local_correlation_branch(query_feat)
            local_support_feat = self.local_correlation_branch(support_feat)
            local_feat = F.conv2d(
                local_query_feat,
                local_support_feat.permute(1, 0, 2, 3),
                groups=2048)
            local_feat = F.relu(local_feat, inplace=True).squeeze(3).squeeze(2)
            local_correlation_cls_score = self.local_correlation_fc_cls(
                local_feat)

        # patch_relation
        if self.patch_relation:
            patch_feat = torch.cat(
                (query_feat, support_feat.expand_as(query_feat)), 1)
            # 7x7 -> 1x1
            patch_feat = self.patch_relation_branch(patch_feat)
            patch_feat = patch_feat.squeeze(3).squeeze(2)
            patch_relation_cls_score = self.patch_relation_fc_cls(patch_feat)

        # aggregate multi relation result
        cls_score_all = patch_relation_cls_score
        if self.local_correlation:
            cls_score_all += local_correlation_cls_score
        if self.global_relation:
            cls_score_all += global_relation_cls_score
        return cls_score_all

    @force_fp32(apply_to=('cls_scores', ))
    def loss(
        self,
        cls_scores: Tensor,
        rois: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        support_gt_labels: Tensor,
        reduction_override: Optional[str] = None
    ) -> Dict:

        losses = dict()
        pdb.set_trace()
        a_mat = assign_label(rois, labels, support_gt_labels)
        if cls_scores is not None:
            if cls_scores.numel() > 0:
                # cls_inds resample the rois to get final classification loss
                losses['loss_cls'] = self.loss_cls(
                    cls_scores[topk_inds],
                    labels[topk_inds],
                    label_weights[topk_inds],
                    avg_factor=len(topk_inds),
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_scores, labels)
        return losses
    
    def cal_IoU_distance_matrix(self, proposal_boxes):
        pairwise_iou = BboxOverlaps2D()
        IoU_matrix = pairwise_iou(proposal_boxes, proposal_boxes)  # .detach().cpu().numpy()
        return IoU_matrix
    
    def _sample_child_nodes(self, center_idx, IoU_matrix):
        # obtain iou array for all the proposals
        act_iou = IoU_matrix[center_idx, :]
        rm_act_iou = act_iou
        rm_act_iou[center_idx] = 0
        pos_iou_idx = torch.where(rm_act_iou > self.iou_threshold)

        if len(pos_iou_idx) != 0:
            pos_iou_arr = rm_act_iou[pos_iou_idx]
            sorted_pos_iou_idx = torch.argsort(-pos_iou_arr)
            selected_pos_iou_idx = sorted_pos_iou_idx.repeat(self.child_iou_num)
            ref_iou_idx = selected_pos_iou_idx[:self.child_iou_num]
            abs_iou_idx = pos_iou_idx[0][ref_iou_idx]
        else:
            abs_iou_idx = torch.tensor([center_idx]).repeat(self.child_iou_num)

        idx = torch.randperm(abs_iou_idx.nelement())
        abs_iou_idx = abs_iou_idx.view(-1)[idx].view(abs_iou_idx.size())
        abs_child_idx = abs_iou_idx[:self.child_num]
        return abs_child_idx
    
    def assign_label(rois, labels, support_gt_labels):
        IoU_matrix = self.cal_IoU_distance_matrix(rois)
        a_mat[:len(rois), :len(rois)] = IoU_matrix
        for i in range(support_gt_labels.size(0)):
            index_ = torch.where(labels == support_gt_labels[i])
            a_mat[index_, i] = 1
            
        
