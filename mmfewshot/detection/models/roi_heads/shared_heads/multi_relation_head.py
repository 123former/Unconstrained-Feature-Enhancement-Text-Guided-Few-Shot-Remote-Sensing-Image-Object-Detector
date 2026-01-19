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
from mmdet.models.builder import HEADS, build_loss

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
                 in_channels = 4096):
        super(MultiRelationHead, self).__init__()
        # following the official implementation patch relation must be True,
        # because only patch relation head contain regression head
        self.patch_relation = True
        self.local_correlation = local_correlation
        self.global_relation = global_relation
        self.iou_threshold = 0.7
        self.in_channels = in_channels
        if self.patch_relation:
            self.patch_relation_fc_cls = nn.Linear(self.in_channels, 1)
        self.loss_cls = build_loss(dict(type='AsymmetricLoss', class_num=143, size_mode=True))


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
       
        if self.patch_relation:
            patch_relation_cls_score = self.patch_relation_fc_cls(torch.cat([query_feat,support_feat], -1))
            
        cls_score_all = patch_relation_cls_score
        return cls_score_all.squeeze()

    def loss(
        self,
        cls_scores: Tensor,
        rois: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        support_gt_labels: Tensor,
        reduction_override: Optional[str] = None) -> Dict:

        losses = dict()
        
        a_mat = self.assign_label(rois, labels, support_gt_labels)

        if cls_scores is not None:
            if cls_scores.numel() > 0:
                # cls_inds resample the rois to get final classification loss
                losses['e_loss'] = self.loss_cls(
                    cls_scores,
                    a_mat,
                    label_weights,
                    avg_factor=None,
                    reduction_override=None)
                # losses['acc'] = accuracy(cls_scores, labels)
        return losses
    
    def assign_label(self, rois, labels, support_gt_labels):
        a_mat = torch.zeros((len(labels)+len(support_gt_labels),len(labels)+len(support_gt_labels)), dtype=labels.dtype).cuda()
        insert_value = torch.ones((len(labels)+len(support_gt_labels),len(labels)+len(support_gt_labels)), dtype=labels.dtype).cuda()
        index_cat = torch.zeros((len(labels)+len(support_gt_labels),len(labels)+len(support_gt_labels)), dtype=labels.dtype).cuda()
        
        for i in range(support_gt_labels.size(0)):
            index_ = torch.where(labels == support_gt_labels[i])
            if len(index_[0]) != 0:
                a_mat[:len(index_[0]), :len(index_[0])] = 1
                a_mat[torch.split(index_[0], 1), len(labels)+i] = 1
                a_mat[len(index_[0]):len(labels), len(index_[0]):len(labels)] = 1
        return a_mat
        
