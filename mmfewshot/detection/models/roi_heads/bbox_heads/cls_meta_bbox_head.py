# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from .bbox_head import BBoxHeadAsy
from torch import Tensor
import pdb


class GRUModel(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            output_size,
    ):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)  # output_size 为输出的维度

    def forward(self, input):
        output, _ = self.gru(input)
        output = self.fc(output[:, -1, :])  # 取最后一个时间步的输出

        return output

@HEADS.register_module()
class ClsMetaBBoxHead(BBoxHeadAsy):
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
                 with_supcls_loss: bool = True,
                 use_meta_sup: bool = True,
                 meta_cls_loss_weight: Optional[float] = None,
                 loss_meta: Dict = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 *args,
                 **kwargs) -> None:

        # kwargs['in_channels'] = 2048
        super().__init__(*args, **kwargs)

        self.with_meta_cls_loss = with_meta_cls_loss
        if with_meta_cls_loss:
            self.fc_meta = nn.Linear(meta_cls_in_channels, num_meta_classes)
            self.meta_cls_loss_weight = meta_cls_loss_weight
            self.loss_meta_cls = build_loss(copy.deepcopy(loss_meta))

        self.with_supcls_loss = with_supcls_loss
        if with_supcls_loss:
            if use_meta_sup:
                self.fc_supcls = nn.Linear(2048, 6)
                self.supcls_loss_weight = meta_cls_loss_weight
                self.loss_supcls = build_loss(copy.deepcopy(loss_meta))
        
            self.fc_supcls_ = nn.Linear(2048, 7)
            self.supcls_loss_weight_ = meta_cls_loss_weight
            self.loss_supcls_ = build_loss(copy.deepcopy(loss_meta))

            # self.fc_sim = nn.Linear(meta_cls_in_channels, 2)
            # self.sim_loss_weight = meta_cls_loss_weight
            # self.loss_sim = build_loss(copy.deepcopy(loss_meta))

            self.gru = GRUModel(input_size=2048, hidden_size=64, num_layers=2, output_size=2048)

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

    def forward_supcls(self, support_feat: Tensor) -> Tensor:
        """Forward function for meta classification.

        Args:
            support_feat (Tensor): Shape of (N, C, H, W).

        Returns:
            Tensor: Box scores with shape of (N, num_meta_classes, H, W).
        """
        supcls_score = self.fc_supcls(support_feat)
        return supcls_score

    @force_fp32(apply_to='supcls_score')
    def loss_supcls_fun(self,
                        supcls_score: Tensor,
                        supcls_labels: Tensor,
                        supcls_label_weights: Tensor,
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
        if self.supcls_loss_weight is None:
            loss_weight = 1. / max(
                torch.sum(supcls_label_weights > 0).float().item(), 1.)
        else:
            loss_weight = self.supcls_loss_weight
        if supcls_score.numel() > 0:
            loss_meta_cls_ = self.loss_supcls(
                supcls_score,
                supcls_labels,
                supcls_label_weights,
                reduction_override=reduction_override)
            losses['loss_supcls'] = loss_meta_cls_ * loss_weight
            losses['sup_acc'] = accuracy(supcls_score, supcls_labels)
        return losses

    def forward_supcls_(self, support_feat: Tensor) -> Tensor:
        """Forward function for meta classification.

        Args:
            support_feat (Tensor): Shape of (N, C, H, W).

        Returns:
            Tensor: Box scores with shape of (N, num_meta_classes, H, W).
        """
        supcls_score = self.fc_supcls_(support_feat)
        return supcls_score

    @force_fp32(apply_to='supcls_score')
    def loss_supcls_fun_(self,
                         supcls_score: Tensor,
                         supcls_labels: Tensor,
                         supcls_label_weights: Tensor,
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
        if self.supcls_loss_weight_ is None:
            loss_weight = 1. / max(
                torch.sum(supcls_label_weights > 0).float().item(), 1.)
        else:
            loss_weight = self.supcls_loss_weight_
        if supcls_score.numel() > 0:
            loss_meta_cls_ = self.loss_supcls_(
                supcls_score,
                supcls_labels,
                supcls_label_weights,
                reduction_override=reduction_override)
            losses['loss_supcls_'] = loss_meta_cls_ * loss_weight
            losses['sup_acc_'] = accuracy(supcls_score, supcls_labels)
        return losses

    @force_fp32(apply_to='sim_score')
    def loss_sim_fun(self,
                     sim_score: Tensor,
                     sim_labels: Tensor,
                     sim_weights: Tensor,
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
        if self.sim_loss_weight is None:
            loss_weight = 1. / max(
                torch.sum(sim_weights > 0).float().item(), 1.)
        else:
            loss_weight = self.sim_loss_weight
        if sim_score.numel() > 0:
            loss_sim_ = self.loss_sim(
                sim_score,
                sim_labels,
                sim_weights,
                reduction_override=reduction_override)
            losses['loss_sim'] = loss_sim_ * loss_weight
        return losses

    # def get_supcls(self, cls_feats, supclass_cls_labels):
    #     use_id = []
    #     sup_cls_feats = []
    #
    #     for supclass_id in supclass_cls_labels:
    #         if supclass_id.item() in use_id:
    #             continue
    #         index_ = torch.where(supclass_cls_labels == supclass_id)
    #         index_split = torch.split(index_[0], 1)
    #         cls_feats_cur = cls_feats[index_split, :]
    #         cls_len = cls_feats_cur.shape[0]
    #         pdb.set_trace()
    #         sup_cls_feats_cur = torch.sum(cls_feats_cur, dim=0)
    #         sup_cls_feats_cur = sup_cls_feats_cur.view(1, -1) / cls_len
    #         use_id.append(supclass_id.item())
    #         sup_cls_feats.append(sup_cls_feats_cur)
    #     sup_cls_feats = torch.cat(sup_cls_feats, 0)
    #
    #     return sup_cls_feats, torch.tensor(use_id).cuda()

    def get_supcls(self, cls_feats, supclass_cls_labels):
        use_id = []
        sup_cls_feats = []

        for supclass_id in supclass_cls_labels:
            if supclass_id.item() in use_id:
                continue
            index_ = torch.where(supclass_cls_labels == supclass_id)
            index_split = torch.split(index_[0], 1)
            cls_feats_cur = cls_feats[index_split, :]
            sup_cls_feats_cur = self.gru(cls_feats_cur.unsqueeze(0))
            use_id.append(supclass_id.item())
            sup_cls_feats.append(sup_cls_feats_cur)
        sup_cls_feats = torch.cat(sup_cls_feats, 0)

        return sup_cls_feats, torch.tensor(use_id).cuda()



