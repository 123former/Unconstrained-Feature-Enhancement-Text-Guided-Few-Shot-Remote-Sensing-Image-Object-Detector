# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Dict, List, Optional

import torch
from mmcv.runner import auto_fp16
from mmcv.utils import ConfigDict
from mmdet.models.builder import DETECTORS
from torch import Tensor
from .gdl import decouple_layer, AffineLayer
from .query_support_detector import QuerySupportDetector
import pdb
from mmfewshot.detection.models.roi_heads.sentence import Text_Embedding
from mmfewshot.detection.models.roi_heads.vis_map import generstae_featmap

@DETECTORS.register_module()
class ClsMetaRCNN(QuerySupportDetector):
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
                 init_cfg: Optional[ConfigDict] = None,
                 use_text=False,
                 init_load=False,
                 init_save_path=None) -> None:
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

        self.is_model_init = False
        self.use_text = use_text
        self.init_load = True
        self.init_save_path = '/home/f523/disk1/sxp/mmfewshot/work_dirs/cls-meta-rcnn_r101_c4_8xb4_voc-split1_base-training_ngnn_plus_augtext_mask'
        # save support template features for model initialization,
        # `_forward_saved_support_dict` used in :func:`forward_model_init`.
        self._forward_saved_support_dict = {
            'gt_labels': [],
            'roi_feats': [],
        }
        self.text_dict = {
            'gt_labels': [],
            'roi_feats': []}
        # save processed support template features for inference,
        # the processed support template features are generated
        # in :func:`model_init`
        self.inference_support_dict = {}
        # self.affine_rpn = AffineLayer(rpn_head['in_channels'], bias=True)
        # self.affine_rcnn = AffineLayer(rpn_head['in_channels'], bias=True)
        # self.RPN_ENABLE_DECOUPLE = False
        # self.ROI_ENABLE_DECOUPLE = False

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
        feats = self.backbone(img, use_meta_conv=True)  #

        # if self.neck is not None:
        #     feats = self.neck(feats)
        if len(feats) != 1:
            return [feats[2]]
        else:
            return feats

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

        query_feats_de_rpn = query_feats
        support_feats_de_rpn = support_feats

        # if self.RPN_ENABLE_DECOUPLE:
        #     scale = 0.0
        #     temp_list = []
        #     for feat in query_feats_de_rpn:
        #         temp_list.append(self.affine_rpn(decouple_layer(feat, scale)))
        #     query_feats_de_rpn = tuple(temp_list)
        #
        #     temp_list = []
        #     for feat in support_feats_de_rpn:
        #         temp_list.append(self.affine_rpn(decouple_layer(feat, scale)))
        #     support_feats_de_rpn = tuple(temp_list)

        query_feats_de_rcnn = query_feats
        support_feats_de_rcnn = support_feats
        # if self.ROI_ENABLE_DECOUPLE:
        #     scale = 0.01
        #     temp_list = []
        #     for feat in query_feats_de_rcnn:
        #         temp_list.append(self.affine_rcnn(decouple_layer(feat, scale)))
        #     query_feats_de_rcnn = tuple(temp_list)
        #
        #     temp_list = []
        #     for feat in support_feats_de_rcnn:
        #         temp_list.append(self.affine_rcnn(decouple_layer(feat, scale)))
        #     support_feats_de_rcnn = tuple(temp_list)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            if self.rpn_with_support:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats_de_rpn,
                    support_feats_de_rpn,
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
                    query_feats_de_rpn,
                    copy.deepcopy(query_data['img_metas']),
                    copy.deepcopy(query_data['gt_bboxes']),
                    gt_labels=None,
                    gt_bboxes_ignore=copy.deepcopy(
                        query_data.get('gt_bboxes_ignore', None)),
                    proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            query_feats_de_rcnn,
            support_feats_de_rcnn,
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

        losses.update(roi_losses)

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
        if self.roi_head.use_text:
            sen_batches = self.roi_head.text_aug.get_sentence_batches(gt_labels)
            roi_feat = [self.roi_head.text_aug.forward(sen_batches, roi_feat[0], fusion=True)]
        if self.roi_head.with_text_sim_aug:
            self._forward_saved_support_dict['gt_labels'].extend(gt_labels)
            self._forward_saved_support_dict['roi_feats'].extend(roi_feat)
            self.text_dict['gt_labels'].extend(gt_labels)
            self.text_dict['roi_feats'].extend([text_feat])
        else:
            self._forward_saved_support_dict['gt_labels'].extend(gt_labels)
            self._forward_saved_support_dict['roi_feats'].extend(roi_feat)
        return {'gt_labels': gt_labels, 'roi_feat': roi_feat}

    def model_init(self):
        """process the saved support features for model initialization."""
        gt_labels = torch.cat(self._forward_saved_support_dict['gt_labels'])
        roi_feats = torch.cat(self._forward_saved_support_dict['roi_feats'])
        # pdb.set_trace()
        # torch.save(gt_labels, 'cls_gt_labels.pkl')
        # torch.save(roi_feats, 'cls_roi_feats.pkl')
        if self.roi_head.with_text_sim_aug:
            text_feats = torch.cat(self.text_dict['roi_feats'])
        class_ids = set(gt_labels.data.tolist())
        self.inference_support_dict.clear()
        for class_id in class_ids:
            if self.roi_head.with_text_sim_aug:
                self.inference_support_dict[class_id] = [roi_feats[gt_labels == class_id].mean([0], True),
                                                         text_feats[gt_labels == class_id].mean([0], True)]
            else:
                self.inference_support_dict[class_id] = roi_feats[
                    gt_labels == class_id].mean([0], True)

        # if self.init_load:
        #     save_path = os.path.join(self.init_save_path, 'base_support_dict.pth')
        #     load_feat = torch.load(save_path)
        #     for key in load_feat.keys():
        #         self.inference_support_dict[key] = copy.deepcopy(load_feat[key])
        # self.inference_support_dict[class_id] = roi_feats[
        #     gt_labels == class_id]
        # set the init flag
        self.is_model_init = True

        if self.init_save_path is not None and not self.init_load:
            save_path = os.path.join(self.init_save_path, 'tune_support_dict.pth')
            # torch.save(self.inference_support_dict, save_path)

        # reset support features buff
        print(class_ids)
        for k in self._forward_saved_support_dict.keys():
            self._forward_saved_support_dict[k].clear()
            self.text_dict[k].clear()

    def forward_test(self, imgs, img_metas, gt_bboxes=None,
                     gt_labels=None, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test_temp(imgs[0], img_metas[0], gt_bboxes=gt_bboxes,
                                         gt_labels=gt_labels, **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    def simple_test_temp(self,
                         img: Tensor,
                         img_metas: List[Dict],
                         proposals: Optional[List[Tensor]] = None,
                         gt_bboxes: Optional[List[Tensor]] = None,
                         gt_labels: Optional[List[Tensor]] = None,
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

        feats = self.backbone(img)
        if self.with_neck:
            query_feats = self.neck(feats)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test(query_feats, img_metas)
        else:
            proposal_list = proposals

        # support_feats = []
        # for key in self.inference_support_dict.keys():
        #     support_feats.append(self.inference_support_dict[key])
        #
        # generstae_featmap(feats[-1], support_feats, img.detach().cpu().numpy(), img_metas[0]['filename'])

        roi_results = self.roi_head.simple_test(
            query_feats,
            copy.deepcopy(self.inference_support_dict),
            proposal_list,
            img_metas,
            gt_labels,
            rescale=rescale)

        return roi_results

    def forward_dummy(self, img):
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
        # assert self.with_bbox, 'Bbox head must be implemented.'
        # assert len(img_metas) == 1, 'Only support single image inference.'
        # if not self.is_model_init:
        #     # process the saved support features
        #     self.model_init()

        query_feats = self.extract_feat(img)
        # if proposals is None:
        #     proposal_list = self.rpn_head.simple_test(query_feats, img_metas)
        # else:
        #     proposal_list = proposals
        #
        # roi_results = self.roi_head.simple_test(
        #     query_feats,
        #     copy.deepcopy(self.inference_support_dict),
        #     proposal_list,
        #     img_metas,
        #     rescale=rescale)

        return query_feats

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
        feats = self.backbone(img)
        if self.with_neck:
            query_feats = self.neck(feats)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test(query_feats, img_metas)
        else:
            proposal_list = proposals

        support_feats = []
        for key in self.inference_support_dict.keys():
            support_feats.append(self.inference_support_dict[key])

        generstae_featmap(feats[-1], support_feats, img.detach().cpu().numpy())
        roi_results = self.roi_head.simple_test(
            query_feats,
            copy.deepcopy(self.inference_support_dict),
            proposal_list,
            img_metas,
            rescale=rescale)

        return roi_results
