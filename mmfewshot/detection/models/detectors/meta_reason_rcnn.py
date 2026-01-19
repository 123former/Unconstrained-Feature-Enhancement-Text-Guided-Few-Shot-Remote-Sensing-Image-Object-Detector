# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmcv.utils import ConfigDict
from mmdet.models.builder import DETECTORS
from torch import Tensor
import pickle
import torch.nn.functional as F
from ..utils import ConvModule
import numpy as np
from mmdet.models.builder import HEADS, build_neck, build_head
import copy

from .query_support_detector import QuerySupportDetector


@DETECTORS.register_module()
class MetaReasonRCNN(QuerySupportDetector):
    """Implementation of `Meta R-CNN.  <https://arxiv.org/abs/1909.13032>`_.

    Args:
        backbone (dict): Config of the backbone for query data.
        neck (dict | None): Config of the neck for query data and
            probably for support data. Default: None.
        support_backbone (dict | None): Config of the backbone for
            support data only. If None, support and query data will
            share same backbone. Default: None.
        support_neck (dict | None): Config of the neck for support
            data only. Default: None.
        rpn_head (dict | None): Config of rpn_head. Default: None.
        roi_head (dict | None): Config of roi_head. Default: None.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        pretrained (str | None): model pretrained path. Default: None.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 backbone: ConfigDict,
                 neck: Optional[ConfigDict] = None,
                 support_backbone: Optional[ConfigDict] = None,
                 support_neck: Optional[ConfigDict] = None,
                 rpn_head: Optional[ConfigDict] = None,
                 roi_head: Optional[ConfigDict] = None,
                 train_cfg: Optional[ConfigDict] = None,
                 test_cfg: Optional[ConfigDict] = None,
                 pretrained: Optional[ConfigDict] = None,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            support_backbone=support_backbone,
            support_neck=support_neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        adj_gt = '/home/f523/guazai/disk3/shangxiping/mmfewshot/tools/graph/new_COCO_graph_a.pkl'  # relation graph: './graph/new_ade_graph_r.pkl'
        graph_out_channels = 256
        normalize = None

        self.is_model_init = False
        # save support template features for model initialization,
        # `_forward_saved_support_dict` used in :func:`forward_model_init`.
        self._forward_saved_support_dict = {
            'gt_labels': [],
            'roi_feats': [],
        }
        # save processed support template features for inference,
        # the processed support template features are generated
        # in :func:`model_init`
        self.inference_support_dict = {}

        roi_head_agg = roi_head.deepcopy()
        if roi_head_agg is not None:
            # update train and test cfg here for now
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            in_channels = roi_head_agg['bbox_head']['in_channels']
            meta_cls_in_channels = roi_head_agg['bbox_head']['meta_cls_in_channels']
            # target_stds = roi_head['bbox_head']['bbox_coder']['target_stds']

            roi_head_agg['bbox_head']['in_channels'] = in_channels + graph_out_channels
            # roi_head_agg['bbox_head']['meta_cls_in_channels'] = meta_cls_in_channels + graph_out_channels
            # roi_head_agg['bbox_head']['bbox_coder']['target_stds'] = [0.05, 0.05, 0.1, 0.1]
            # if rcnn_train_cfg is not None:
            #     rcnn_train_cfg['assigner']['pos_iou_thr'] = 0.6
            #     rcnn_train_cfg['assigner']['neg_iou_thr'] = 0.6
            #     rcnn_train_cfg['assigner']['min_pos_iou'] = 0.6

            roi_head_agg.update(train_cfg=rcnn_train_cfg)
            roi_head_agg.update(test_cfg=test_cfg.rcnn)
            roi_head_agg.pretrained = pretrained

            agg_in_channel = roi_head_agg['aggregation_layer']['aggregator_cfgs'][0]['in_channels']
            roi_head_agg['aggregation_layer']['aggregator_cfgs'][0]['in_channels'] = agg_in_channel + graph_out_channels

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
            nn.Linear(1024 // 16, roi_head['bbox_head']['in_channels'] + 1))
        # init graph w
        self.graph_out_channels = graph_out_channels
        self.graph_weight_fc = nn.Linear(roi_head['bbox_head']['in_channels'] + 1, self.graph_out_channels)
        self.relu = nn.ReLU(inplace=True)

    @auto_fp16(apply_to=('img',))
    def extract_support_feat(self, img):
        """Extracting features from support data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of input image, each item with shape
                (N, C, H, W).
        """
        feats = self.backbone(img, use_meta_conv=True)
        if self.support_neck is not None:
            feats = self.support_neck(feats)
        return feats

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

    def get_enhanced_feat(self, base_feat, img_meta, cls_score):
        # add reasoning process
        bbox_head = self.roi_head.bbox_head

        # 1.build global semantic pool
        global_semantic_pool = torch.cat((bbox_head.fc_cls.weight,
                                          bbox_head.fc_cls.bias.unsqueeze(1)), 1).detach()

        # 2.compute graph attention
        attention_map = nn.Softmax(1)(torch.mm(base_feat, torch.transpose(global_semantic_pool, 0, 1)))
        # 3.adaptive global reasoning
        temp = self.adj_gt * torch.eye(81).cuda()

        alpha_em = attention_map.unsqueeze(-1) * torch.mm(temp, global_semantic_pool).unsqueeze(0)
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
                      query_data: Dict,
                      support_data: Dict,
                      proposals: Optional[List] = None,
                      **kwargs) -> Dict:
        """Forward function for training.

        Args:
            query_data (dict): In most cases, dict of query data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            support_data (dict):  In most cases, dict of support data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            proposals (list): Override rpn proposals with custom proposals.
                Use when `with_rpn` is False. Default: None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        query_img = query_data['img']
        support_img = support_data['img']
        query_feats = self.extract_query_feat(query_img)
        support_feats = self.extract_support_feat(support_img)

        losses = dict()

        base_feat_query = self.get_base_feat(query_feats)
        # base_feat_support = self.get_base_feat(support_feats)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            if self.rpn_with_support:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats,
                    support_feats,
                    query_img_metas=query_data['img_metas'],
                    query_gt_bboxes=query_data['gt_bboxes'],
                    query_gt_labels=None,
                    query_gt_bboxes_ignore=query_data.get(
                        'gt_bboxes_ignore', None),
                    support_img_metas=support_data['img_metas'],
                    support_gt_bboxes=support_data['gt_bboxes'],
                    support_gt_labels=support_data['gt_labels'],
                    support_gt_bboxes_ignore=support_data.get(
                        'gt_bboxes_ignore', None),
                    proposal_cfg=proposal_cfg)
            else:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats,
                    copy.deepcopy(query_data['img_metas']),
                    copy.deepcopy(query_data['gt_bboxes']),
                    gt_labels=None,
                    gt_bboxes_ignore=copy.deepcopy(
                        query_data.get('gt_bboxes_ignore', None)),
                    proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses, roi_results = self.roi_head.forward_train(
            query_feats,
            support_feats,
            proposals=proposal_list,
            query_img_metas=query_data['img_metas'],
            query_gt_bboxes=query_data['gt_bboxes'],
            query_gt_labels=query_data['gt_labels'],
            query_gt_bboxes_ignore=query_data.get('gt_bboxes_ignore', None),
            support_img_metas=support_data['img_metas'],
            support_gt_bboxes=support_data['gt_bboxes'],
            support_gt_labels=support_data['gt_labels'],
            support_gt_bboxes_ignore=support_data.get('gt_bboxes_ignore', None),
            return_results=True,
            **kwargs)

        losses.update(roi_losses)

        cls_score = roi_results['cls_score']

        # temp_tensor = torch.eye(cls_score)

        enhanced_feat_query = self.get_enhanced_feat(base_feat_query, query_data['img_metas'], cls_score)
        self.roi_head_agg.aug_feat = enhanced_feat_query
        # enhanced_feat_support = self.get_enhanced_feat(base_feat_support, support_data['img_metas'], cls_score)
        # bbox_feats = torch.cat([bbox_feats, enhanced_feat_query], 1)

        roi_agg_losses = self.roi_head_agg.forward_train(
            query_feats,
            support_feats,
            proposals=proposal_list,
            query_img_metas=query_data['img_metas'],
            query_gt_bboxes=query_data['gt_bboxes'],
            query_gt_labels=query_data['gt_labels'],
            query_gt_bboxes_ignore=query_data.get('gt_bboxes_ignore', None),
            support_img_metas=support_data['img_metas'],
            support_gt_bboxes=support_data['gt_bboxes'],
            support_gt_labels=support_data['gt_labels'],
            support_gt_bboxes_ignore=support_data.get('gt_bboxes_ignore',
                                                      None),
            **kwargs)

        for key in roi_agg_losses.keys():
            new_key = key + '_agg'
            losses[new_key] = roi_agg_losses[key]

        return losses

    def forward_model_init(self,
                           img: Tensor,
                           img_metas: List[Dict],
                           gt_bboxes: List[Tensor] = None,
                           gt_labels: List[Tensor] = None,
                           **kwargs):
        """extract and save support features for model initialization.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.

        Returns:
            dict: A dict contains following keys:

                - `gt_labels` (Tensor): class indices corresponding to each
                    feature.
                - `res5_rois` (list[Tensor]): roi features of res5 layer.
        """
        # `is_model_init` flag will be reset when forward new data.
        self.is_model_init = False
        assert len(gt_labels) == img.size(
            0), 'Support instance have more than two labels'
        feats = self.extract_support_feat(img)
        roi_feat = self.roi_head.extract_support_feats(feats)
        self._forward_saved_support_dict['gt_labels'].extend(gt_labels)
        self._forward_saved_support_dict['roi_feats'].extend(roi_feat)
        return {'gt_labels': gt_labels, 'roi_feat': roi_feat}

    def model_init(self):
        """process the saved support features for model initialization."""
        gt_labels = torch.cat(self._forward_saved_support_dict['gt_labels'])
        roi_feats = torch.cat(self._forward_saved_support_dict['roi_feats'])
        class_ids = set(gt_labels.data.tolist())
        self.inference_support_dict.clear()
        for class_id in class_ids:
            self.inference_support_dict[class_id] = roi_feats[
                gt_labels == class_id].mean([0], True)
        # set the init flag
        self.is_model_init = True
        # reset support features buff
        for k in self._forward_saved_support_dict.keys():
            self._forward_saved_support_dict[k].clear()

    def simple_test(self,
                    img: Tensor,
                    img_metas: List[Dict],
                    proposals: Optional[List[Tensor]] = None,
                    rescale: bool = False):
        """Test without augmentation.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor] | None): override rpn proposals with
                custom proposals. Use when `with_rpn` is False. Default: None.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) == 1, 'Only support single image inference.'
        if not self.is_model_init:
            # process the saved support features
            self.model_init()

        query_feats = self.extract_feat(img)
        base_feat_query = self.get_base_feat(query_feats)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test(query_feats, img_metas)
        else:
            proposal_list = proposals

        roi_results = self.roi_head.simple_test(query_feats,
                                                copy.deepcopy(self.inference_support_dict),
                                                proposal_list,
                                                img_metas,
                                                rescale=rescale)

        cls_score = self.roi_head.bbox_results['cls_score']

        enhanced_feat_query = self.get_enhanced_feat(base_feat_query, img_metas, cls_score)
        self.roi_head_agg.aug_feat = enhanced_feat_query

        roi_results_agg = self.roi_head_agg.simple_test(query_feats,
                                                        copy.deepcopy(self.inference_support_dict),
                                                        proposal_list,
                                                        img_metas,
                                                        rescale=rescale)

        return roi_results_agg
