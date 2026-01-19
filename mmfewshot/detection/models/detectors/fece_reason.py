# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS, HEADS, build_neck, build_head
from mmdet.models.detectors.two_stage import TwoStageDetector
import pdb
from torch import Tensor
import pickle
import torch.nn.functional as F
from ..utils import ConvModule
import numpy as np
import copy
import torch
import torch.nn as nn


@DETECTORS.register_module()
class FSCER(TwoStageDetector):
    """Implementation of `FSCE <https://arxiv.org/abs/2103.05950>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 adj_gt=None):
        super(FSCER, self).__init__(backbone=backbone,
                                    neck=neck,
                                    rpn_head=rpn_head,
                                    roi_head=roi_head,
                                    train_cfg=train_cfg,
                                    test_cfg=test_cfg,
                                    pretrained=pretrained,
                                    init_cfg=init_cfg)
        adj_gt = adj_gt  # relation graph: './graph/new_ade_graph_r.pkl'
        graph_out_channels = 256
        normalize = None

        roi_head_agg = roi_head.deepcopy()
        if roi_head_agg is not None:
            # update train and test cfg here for now
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head_agg['bbox_head']['add_channel'] = True
            if self.roi_head.Cascade:
                roi_head_agg['bbox_head']['bbox_coder']['target_stds'] = [0.05, 0.05, 0.1, 0.1]
                if rcnn_train_cfg is not None:
                    rcnn_train_cfg['assigner']['pos_iou_thr'] = 0.6
                    rcnn_train_cfg['assigner']['neg_iou_thr'] = 0.6
                    rcnn_train_cfg['assigner']['min_pos_iou'] = 0.6

            roi_head_agg.update(train_cfg=rcnn_train_cfg)
            roi_head_agg.update(test_cfg=test_cfg.rcnn)
            roi_head_agg.pretrained = pretrained

            self.roi_head_agg = build_head(roi_head_agg)

        self.normalize = normalize
        self.with_bias = normalize is None

        if adj_gt is not None:
            self.adj_gt = pickle.load(open(adj_gt, 'rb'))
            self.adj_gt = np.float32(self.adj_gt)
            self.adj_gt = nn.Parameter(torch.from_numpy(self.adj_gt), requires_grad=False)
        # init cmp attention
        self.cmp_attention = nn.ModuleList()
        self.cmp_attention.append(
            ConvModule(1024, 1024 // 16,
                       3, stride=2, padding=1, normalize=self.normalize, bias=self.with_bias))
        self.cmp_attention.append(
            nn.Linear(1024 // 16, roi_head['bbox_head']['fc_out_channels']))
        # init graph w
        self.graph_out_channels = graph_out_channels
        self.graph_weight_fc = nn.Linear(roi_head['bbox_head']['fc_out_channels'], self.graph_out_channels)
        self.relu = nn.ReLU(inplace=True)

    def get_base_feat(self, x):
        # precmp attention
        if len(x) > 1:
            base_feat = []
            for b_f in x[1:]:
                base_feat.append(
                    F.interpolate(b_f, scale_factor=(x[2].size(2) / b_f.size(2), x[2].size(3) / b_f.size(3))))
            base_feat = torch.cat(base_feat, 1)
        else:
            base_feat = torch.cat(x, 1)

        for ops in self.cmp_attention:
            base_feat = ops(base_feat)
            if len(base_feat.size()) > 2:
                base_feat = base_feat.mean(3).mean(2)
            else:
                base_feat = self.relu(base_feat)
        return base_feat

    def get_enhanced_feat(self, base_feat, img_meta, cls_score, cls=True):
        # add reasoning process
        bbox_head = self.roi_head.bbox_head

        # 1.build global semantic pool
        if cls:
            global_semantic_pool = (bbox_head.fc_cls.weight).detach()
        else:
            global_semantic_pool = (bbox_head.fc_reg.weight).detach()
        # pdb.set_trace()
        # 2.compute graph attention
        attention_map = nn.Softmax(1)(torch.mm(base_feat, torch.transpose(global_semantic_pool, 0, 1)))
        # 3.adaptive global reasoning

        alpha_em = attention_map.unsqueeze(-1) * torch.mm(self.adj_gt, global_semantic_pool).unsqueeze(0)
        alpha_em = alpha_em.view(-1, global_semantic_pool.size(-1))
        alpha_em = self.graph_weight_fc(alpha_em)
        alpha_em = self.relu(alpha_em)
        # enhanced_feat = torch.mm(nn.Softmax(1)(cls_score), alpha_em)
        n_classes = bbox_head.fc_cls.weight.size(0)
        cls_prob = nn.Softmax(1)(cls_score).view(len(img_meta), -1, n_classes)

        enhanced_feat = torch.bmm(cls_prob, alpha_em.view(len(img_meta), -1, self.graph_out_channels))
        enhanced_feat = enhanced_feat.view(-1, self.graph_out_channels)
        # temp = torch.ones_like(enhanced_feat)
        return enhanced_feat

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        base_feat = self.get_base_feat(x)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        cls_score = self.roi_head.bbox_results['cls_score']

        if self.roi_head.Cascade:
            proposal_list = self.roi_head.proposal_list

        enhanced_feat_cls = self.get_enhanced_feat(base_feat, img_metas, cls_score)
        self.roi_head_agg.bbox_head.aug_feat_cls = enhanced_feat_cls

        # enhanced_feat_reg = self.get_enhanced_feat(base_feat, img_metas, cls_score, cls=False)
        # self.roi_head_agg.bbox_head.aug_feat_reg = enhanced_feat_cls

        roi_agg_losses = self.roi_head_agg.forward_train(x, img_metas, proposal_list,
                                                         gt_bboxes, gt_labels,
                                                         gt_bboxes_ignore, gt_masks,
                                                         **kwargs)

        for key in roi_agg_losses.keys():
            new_key = key + '_agg'
            losses[new_key] = roi_agg_losses[key]
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        base_feat = self.get_base_feat(x)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        result1 = self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

        cls_score = self.roi_head.bbox_results['cls_score']

        enhanced_feat_cls = self.get_enhanced_feat(base_feat, img_metas, cls_score)
        self.roi_head_agg.bbox_head.aug_feat_cls = enhanced_feat_cls

        # enhanced_feat_reg = self.get_enhanced_feat(base_feat, img_metas, cls_score, cls=False)
        # self.roi_head_agg.bbox_head.aug_feat_reg = enhanced_feat_cls

        result2 = self.roi_head_agg.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

        return result2
