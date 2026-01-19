# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads import BBoxHead, Shared2FCBBoxHead
from torch import Tensor
from mmcv.cnn import build_conv_layer, ConvModule


@HEADS.register_module()
class ReasonBBoxHead(Shared2FCBBoxHead):
    """BBoxHead with meta classification for metarcnn and fsdetview.

    Args:
        num_meta_classes (int): Number of classes for meta classification.
        meta_cls_in_channels (int): Number of support feature channels.
        with_meta_cls_loss (bool): Use meta classification loss.
            Default: True.
        meta_cls_loss_weight (float | None): The loss weight of `loss_meta`.
            Default: None.
        loss_meta (dict): Config for meta classification loss.
    """

    def __init__(self,
                 num_meta_classes: int,
                 meta_cls_in_channels: int = 2048,
                 with_meta_cls_loss: bool = True,
                 meta_cls_loss_weight: Optional[float] = None,
                 with_reason_cl_loss: bool = True,
                 with_multi_conv: bool = True,
                 loss_meta: Dict = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 frozen=False,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        conv_cfg = dict(type='Conv2d')
        norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
        act_cfg = dict(type='Swish')

        self.frozen = frozen

        self.with_meta_cls_loss = with_meta_cls_loss
        if with_meta_cls_loss:
            self.fc_meta = nn.Linear(meta_cls_in_channels, num_meta_classes)
            self.meta_cls_loss_weight = meta_cls_loss_weight
            self.loss_meta_cls = build_loss(copy.deepcopy(loss_meta))
        self.with_reason_cls_loss = with_reason_cl_loss
        if self.with_reason_cls_loss:
            # supclass
            self.fc_supclass = nn.Linear(meta_cls_in_channels, 14)
            self.supclass_cls_loss_weight = meta_cls_loss_weight
            self.loss_supclass_cls = build_loss(copy.deepcopy(loss_meta))

            # man_made
            self.fc_man_made = nn.Linear(meta_cls_in_channels, 3)
            self.man_made_cls_loss_weight = meta_cls_loss_weight
            self.loss_man_made_cls = build_loss(copy.deepcopy(loss_meta))

            # envir
            self.fc_envir = nn.Linear(meta_cls_in_channels, 4)
            self.envir_cls_loss_weight = meta_cls_loss_weight
            self.loss_envir_cls = build_loss(copy.deepcopy(loss_meta))

        self.with_multi_conv = with_multi_conv
        if self.with_multi_conv:
            self.uncertain_layers = self.init_layer(conv_cfg, norm_cfg, act_cfg)
            self.supclass_layers = self.init_layer(conv_cfg, norm_cfg, act_cfg)
            self.man_made_layers = self.init_layer(conv_cfg, norm_cfg, act_cfg)
            self.envir_layers = self.init_layer(conv_cfg, norm_cfg, act_cfg)

            self.fusion_layers = nn.Sequential(
                nn.Conv2d(
                    1024 * 4,
                    1024 - 256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                nn.ReLU(inplace=True))

            self.support_layer = nn.Sequential(
                nn.Conv2d(
                    1024,
                    256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                nn.ReLU(inplace=True))
        self.freeze_p()
    def init_layer(self, conv_cfg, norm_cfg, act_cfg):
        layer = nn.Sequential(
            nn.Conv2d(
                256,
                1024,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.ReLU(inplace=True))
        return layer

    def freeze_p(self):
        if self.frozen:
            self.support_layer.eval()
            for param in self.support_layer.parameters():
                param.requires_grad = False

            self.uncertain_layers.eval()
            for param in self.uncertain_layers.parameters():
                param.requires_grad = False

            self.supclass_layers.eval()
            for param in self.supclass_layers.parameters():
                param.requires_grad = False

            self.man_made_layers.eval()
            for param in self.man_made_layers.parameters():
                param.requires_grad = False

            self.envir_layers.eval()
            for param in self.envir_layers.parameters():
                param.requires_grad = False

            self.fusion_layers.eval()
            for param in self.fusion_layers.parameters():
                param.requires_grad = False
            
            self.fc_supclass.eval()
            for param in self.fc_supclass.parameters():
                param.requires_grad = False
                
            self.fc_man_made.eval()
            for param in self.fc_man_made.parameters():
                param.requires_grad = False
                
            self.fc_envir.eval()
            for param in self.fc_envir.parameters():
                param.requires_grad = False
                
            
    def forward_muli_conv(self, feat, support=False):
        if support:
            feat = self.support_layer(feat)
        uncertain_feat = self.uncertain_layers(feat)
        supclass_feat = self.supclass_layers(feat)
        man_made_feat = self.man_made_layers(feat)
        envir_feat = self.envir_layers(feat)
        multi_feat = [uncertain_feat, supclass_feat, man_made_feat, envir_feat]
        fusion_feat = self.fusion_layers(torch.cat(multi_feat, 1))
        fusion_feat = torch.cat([fusion_feat, feat], 1)

        return [uncertain_feat, supclass_feat, man_made_feat, envir_feat, fusion_feat]

    def forward_meta_cls(self, support_feat: Tensor) -> Tensor:
        """Forward function for meta classification.

        Args:
            support_feat (Tensor): Shape of (N, C, H, W).

        Returns:
            Tensor: Box scores with shape of (N, num_meta_classes, H, W).
        """
        meta_cls_score = self.fc_meta(support_feat)
        return meta_cls_score

    @force_fp32(apply_to='meta_cls_score')
    def loss_meta(self,
                  meta_cls_score: Tensor,
                  meta_cls_labels: Tensor,
                  meta_cls_label_weights: Tensor,
                  reduction_override: Optional[str] = None) -> Dict:
        """Meta classification loss.

        Args:
            meta_cls_score (Tensor): Predicted meta classification scores
                 with shape (N, num_meta_classes).
            meta_cls_labels (Tensor): Corresponding class indices with
                shape (N).
            meta_cls_label_weights (Tensor): Meta classification loss weight
                of each sample with shape (N).
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss. Options
                are "none", "mean" and "sum". Default: None.

        Returns:
            Dict: The calculated loss.
        """
        losses = dict()
        if self.meta_cls_loss_weight is None:
            loss_weight = 1. / max(
                torch.sum(meta_cls_label_weights > 0).float().item(), 1.)
        else:
            loss_weight = self.meta_cls_loss_weight

        if meta_cls_score.numel() > 0:
            loss_meta_cls_ = self.loss_meta_cls(
                meta_cls_score,
                meta_cls_labels,
                meta_cls_label_weights,
                reduction_override=reduction_override)
            losses['loss_meta_cls'] = loss_meta_cls_ * loss_weight
            losses['meta_acc'] = accuracy(meta_cls_score, meta_cls_labels)
        return losses

    def forward_reason_cls(self, feats) -> Tensor:
        """Forward function for meta classification.

        Args:
            feats (Tensor): Shape of (N, C, H, W).

        Returns:
            Tensor: Box scores with shape of (N, num_meta_classes, H, W).
        """
        supclass_cls_score = self.fc_supclass(feats[1])
        man_made_cls_score = self.fc_man_made(feats[2])
        envir_cls_score = self.fc_envir(feats[3])
        reason_cls_score = dict(supclass_cls_score=supclass_cls_score,
                                man_made_cls_score=man_made_cls_score,
                                envir_cls_score=envir_cls_score
                                )
        return reason_cls_score

    @force_fp32(apply_to='reason_cls_score')
    def loss_reason(self,
                    reason_cls_score: Dict,
                    reason_cls_labels: Dict,
                    reason_cls_label_weights: Tensor,
                    reduction_override: Optional[str] = None,
                    flag='support') -> Dict:
        """Meta classification loss.

        Args:
            meta_cls_score (Tensor): Predicted meta classification scores
                 with shape (N, num_meta_classes).
            meta_cls_labels (Tensor): Corresponding class indices with
                shape (N).
            meta_cls_label_weights (Tensor): Meta classification loss weight
                of each sample with shape (N).
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss. Options
                are "none", "mean" and "sum". Default: None.

        Returns:
            Dict: The calculated loss.
        """
        losses = dict()
        loss_reason_cls = 0
        loss_reason_acc = 0
        if self.supclass_cls_loss_weight is None:
            supclass_cls_loss_weight = 1. / max(
                torch.sum(reason_cls_label_weights > 0).float().item(), 1.)
        else:
            supclass_cls_loss_weight = reason_cls_label_weights

        if self.man_made_cls_loss_weight is None:
            man_made_cls_loss_weight = 1. / max(
                torch.sum(reason_cls_label_weights > 0).float().item(), 1.)
        else:
            man_made_cls_loss_weight = reason_cls_label_weights

        if self.envir_cls_loss_weight is None:
            envir_cls_loss_weight = 1. / max(
                torch.sum(reason_cls_label_weights > 0).float().item(), 1.)
        else:
            envir_cls_loss_weight = reason_cls_label_weights

        if reason_cls_score['supclass_cls_score'].numel() > 0:
            loss_supclass_cls = self.loss_meta_cls(
                reason_cls_score['supclass_cls_score'],
                reason_cls_labels['gt_supclass_cls'][:, 0] - 100,
                reason_cls_label_weights,
                reduction_override=reduction_override)
            loss_reason_cls = loss_reason_cls + loss_supclass_cls * supclass_cls_loss_weight
            loss_reason_acc = loss_reason_acc + accuracy(reason_cls_score['supclass_cls_score'],
                                                         reason_cls_labels['gt_supclass_cls'][:, 0] - 100)

        if reason_cls_score['man_made_cls_score'].numel() > 0:
            loss_made_cls = self.loss_meta_cls(
                reason_cls_score['man_made_cls_score'],
                reason_cls_labels['gt_man_made_cls'][:, 0] - 200,
                reason_cls_label_weights,
                reduction_override=reduction_override)
            loss_reason_cls = loss_reason_cls + loss_made_cls * man_made_cls_loss_weight
            loss_reason_acc = loss_reason_acc + accuracy(reason_cls_score['man_made_cls_score'],
                                                         reason_cls_labels['gt_man_made_cls'][:, 0] - 200)

        if reason_cls_score['envir_cls_score'].numel() > 0:
            loss_envir_cls = self.loss_meta_cls(
                reason_cls_score['envir_cls_score'],
                reason_cls_labels['gt_envir_cls'][:, 0] - 300,
                reason_cls_label_weights,
                reduction_override=reduction_override)
            loss_reason_cls = loss_reason_cls + loss_envir_cls * envir_cls_loss_weight
            loss_reason_acc = loss_reason_acc + accuracy(reason_cls_score['envir_cls_score'],
                                                         reason_cls_labels['gt_envir_cls'][:, 0] - 300)
        if flag == 'support':
            losses['loss_reason_meta_cls'] = loss_reason_cls / 3
            losses['reason_meta_acc'] = loss_reason_acc / 3
        else:
            losses['loss_reason_cls'] = loss_reason_cls / 3
            losses['reason_acc'] = loss_reason_acc / 3

        return losses
