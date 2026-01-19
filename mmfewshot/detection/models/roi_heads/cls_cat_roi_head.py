from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from .vfa_roi_head import VFARoIHead

@HEADS.register_module()
class ClsRoIHead(VFARoIHead):
    def _bbox_forward_train_cat(self, query_feats: List[Tensor],
                                support_feats: List[Tensor],
                                sampling_results: object,
                                query_img_metas: List[Dict],
                                query_gt_bboxes: List[Tensor],
                                query_gt_labels: List[Tensor],
                                support_gt_labels: List[Tensor]) -> Dict:
        """Forward function and calculate loss for box head in training.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            sampling_results (obj:`SamplingResult`): Sampling results.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each query
                image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y]
                format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images.
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images.

        Returns:
            dict: Predicted results and losses.
        """
        query_rois = bbox2roi([res.bboxes for res in sampling_results])
        query_roi_feats = self.extract_query_roi_feat(query_feats, query_rois)

        support_feat = self.extract_support_feats(support_feats)[0]

        bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                  query_gt_bboxes,
                                                  query_gt_labels,
                                                  self.train_cfg)
        (labels, label_weights, bbox_targets, bbox_weights) = bbox_targets

        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': [], }
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size
        bbox_results = None
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            cls_scores = []
            bbox_preds = []
            background = []
            support_feat_, query_roi_feats_ = support_feat, query_roi_feats[start:end]

            # sim_rate = torch.cosine_similarity(support_feat[16].unsqueeze(0), query_roi_feats[start:end], 1)
            for i in range(support_feat_.size(0)):
                # pdb.set_trace()

                bbox_results = self._bbox_forward(
                    query_roi_feats_,
                    support_feat_[i].unsqueeze(0))
                # bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)
                cls_scores.append(bbox_results['cls_score'][:, i:i + 1])
                bbox_preds.append(bbox_results['bbox_pred'][:, i * 4:(i + 1) * 4])
                background.append(bbox_results['cls_score'][:, -1:])

            background = torch.mean(torch.cat(background, 1), dim=1).view(background[0].shape[0], -1)
            cls_scores.append(background)
            cls_scores = torch.cat(cls_scores, 1)
            bbox_preds = torch.cat(bbox_preds, 1)
            single_loss_bbox = self.bbox_head.loss(
                cls_scores, bbox_preds,
                query_rois[start:end], labels[start:end],
                label_weights[start:end], bbox_targets[start:end],
                bbox_weights[start:end])
            for key in single_loss_bbox.keys():
                loss_bbox[key].append(single_loss_bbox[key])

        if bbox_results is not None:
            for key in loss_bbox.keys():
                if 'acc' in key:
                    loss_bbox[key] = torch.cat(loss_bbox[key]).mean()
                else:
                    loss_bbox[key] = torch.stack(loss_bbox[key]).sum() / batch_size

        # meta classification loss
        if self.bbox_head.with_meta_cls_loss:
            support_feat_, query_roi_feats_ = support_feat, query_roi_feats[start:end]
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat_)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
