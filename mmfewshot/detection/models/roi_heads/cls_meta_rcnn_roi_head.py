# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmcv.utils import ConfigDict
from mmdet.core import bbox2result, bbox2roi
from mmdet.models.builder import HEADS, build_neck
from mmdet.models.roi_heads import StandardRoIHead
from torch import Tensor
from .multi_case import coco_reason_cls, COCO_SPLIT, voc_reason_cls, VOC_SPLIT
import torch.nn.functional as F
import pdb
from .shared_heads import GNN
from .bbox_heads import decouple_layer, AffineLayer
from .sentence import Text_Embedding
from .vis_map import generstae_featmap

@HEADS.register_module()
class ClsMetaRCNNRoIHead(StandardRoIHead):
    """Roi head for `MetaRCNN <https://arxiv.org/abs/1909.13032>`_.

    Args:
        aggregation_layer (ConfigDict): Config of `aggregation_layer`.
            Default: None.
    """

    def __init__(self,
                 aggregation_layer: Optional[ConfigDict] = None,
                 classes='ALL_CLASSES_SPLIT1',
                 use_gnn=False,
                 use_meta_sup=True,
                 with_sim_loss=False,
                 flag=3,
                 use_text=False,
                 with_text_sim_loss=False,
                 with_text_sim_aug=False,
                 text_output_size=2048,
                 mask_enable=False,
                 constrain=False,
                 use_dior=True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if flag == 4:
            if classes in ['ALL_CLASSES', 'NOVEL_CLASSES', 'BASE_CLASSES']:
                self.classes = COCO_SPLIT[classes]
                self.reason_cls = coco_reason_cls
            else:
                self.classes = VOC_SPLIT[classes]
                self.reason_cls = voc_reason_cls
        else:
            self.classes = None
            self.reason_cls = None

        self.use_text = use_text

        assert aggregation_layer is not None, \
            'missing config of `aggregation_layer`.'
        self.aggregation_layer = build_neck(copy.deepcopy(aggregation_layer))
        self.use_gnn = use_gnn
        self.use_meta_sup = use_meta_sup
        self.flag = flag
        self.with_sim_loss = with_sim_loss
        self.with_text_sim_loss = with_text_sim_loss
        self.with_text_sim_aug = with_text_sim_aug
        if use_gnn:
            self.gnn = GNN()
            # self.affine_gnn = AffineLayer(4096, bias=True, reduce_channels=True)
            self.GNN_ENABLE = False
        if self.use_text:
            self.text_aug = Text_Embedding(output_size=text_output_size, use_dior=use_dior, SPLIT=classes,
                                           mask_enable=mask_enable,
                                           constrain=constrain)

    def forward_train(self,
                      query_feats: List[Tensor],
                      support_feats: List[Tensor],
                      proposals: List[Tensor],
                      query_img_metas: List[Dict],
                      query_gt_bboxes: List[Tensor],
                      query_gt_labels: List[Tensor],
                      support_gt_labels: List[Tensor],
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      return_results=False,
                      **kwargs) -> Dict:
        """Forward function for training.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            proposals (list[Tensor]): List of region proposals with positive
                and negative pairs.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each
                query image, each item with shape (num_gts, 4)
                in [tl_x, tl_y, br_x, br_y] format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images, each item with shape (num_gts).
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images, each item with shape (1).
            query_gt_bboxes_ignore (list[Tensor] | None): Specify which
                bounding boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """

        # assign gts and sample proposals
        sampling_results = []

        if self.with_bbox:
            num_imgs = len(query_img_metas)
            if query_gt_bboxes_ignore is None:
                query_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposals[i], query_gt_bboxes[i],
                    query_gt_bboxes_ignore[i], query_gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposals[i],
                    query_gt_bboxes[i],
                    query_gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in query_feats])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            flag = self.flag
            if flag == 1:  # 1 
                bbox_results = self._bbox_forward_train(
                    query_feats, support_feats, sampling_results, query_img_metas,
                    query_gt_bboxes, query_gt_labels, support_gt_labels)
                if bbox_results is not None:
                    losses.update(bbox_results['loss_bbox'])
            elif flag == 2:  # 2 score
                bbox_results = self._bbox_forward_train_score(
                    query_feats, support_feats, sampling_results, query_img_metas,
                    query_gt_bboxes, query_gt_labels, support_gt_labels)
                if bbox_results is not None:
                    losses.update(bbox_results['loss_bbox'])

            elif flag == 3:  # 3 cat sup
                bbox_results = self._bbox_forward_train_cat_sup(
                    query_feats, support_feats, sampling_results, query_img_metas,
                    query_gt_bboxes, query_gt_labels, support_gt_labels)
                if bbox_results is not None:
                    losses.update(bbox_results['loss_bbox'])
            elif flag == 4:  # 4 2cat sup
                bbox_results = self._bbox_forward_train_2cat_sup(
                    query_feats, support_feats, sampling_results, query_img_metas,
                    query_gt_bboxes, query_gt_labels, support_gt_labels)
                if bbox_results is not None:
                    losses.update(bbox_results['loss_bbox'])
            elif flag == 5:  # 5 cat normal sup no_support
                bbox_results = self._bbox_forward_train_cat_norsup(
                    query_feats, support_feats, sampling_results, query_img_metas,
                    query_gt_bboxes, query_gt_labels, support_gt_labels)
                if bbox_results is not None:
                    losses.update(bbox_results['loss_bbox'])
            elif flag == 6:  # 6 cat normal sup no_support asyloss
                bbox_results = self._bbox_forward_train_2cat_sup_asy(
                    query_feats, support_feats, sampling_results, query_img_metas,
                    query_gt_bboxes, query_gt_labels, support_gt_labels)
                if bbox_results is not None:
                    losses.update(bbox_results['loss_bbox'])
            elif flag == 7:  # cat
                bbox_results = self._bbox_forward_train_cat(
                    query_feats, support_feats, sampling_results, query_img_metas,
                    query_gt_bboxes, query_gt_labels, support_gt_labels)
                if bbox_results is not None:
                    losses.update(bbox_results['loss_bbox'])
            elif flag == 8:  # cat
                bbox_results = self._bbox_forward_train_cat_text(
                    query_feats, support_feats, sampling_results, query_img_metas,
                    query_gt_bboxes, query_gt_labels, support_gt_labels)
                if bbox_results is not None:
                    losses.update(bbox_results['loss_bbox'])
            else:
                bbox_results = self._bbox_forward_train_text(
                    query_feats, support_feats, sampling_results, query_img_metas,
                    query_gt_bboxes, query_gt_labels, support_gt_labels)
                if bbox_results is not None:
                    losses.update(bbox_results['loss_bbox'])
        if return_results:
            return losses, bbox_results
        else:
            return losses

    def sim_loss(self, support_feats, query_roi_feats, query_labels, support_labels):
        support_labels = torch.cat(support_labels)
        num_class = self.bbox_head.num_classes
        index_b = torch.where(query_labels == num_class)
        index_split_b = torch.split(index_b[0], 1)
        roi_feat_b = torch.mean(query_roi_feats[index_split_b, :], dim=0).view(1, -1)

        index_f = torch.where(query_labels != num_class)
        index_split_f = torch.split(index_f[0], 1)
        roi_feat_f = query_roi_feats[index_split_f, :]
        f_labels = query_labels[index_f[0]]

        total_feats = torch.cat([roi_feat_b, roi_feat_f, support_feats])
        total_label = torch.cat([torch.tensor([num_class]).cuda(), f_labels, support_labels])
        dot_product_mat = torch.mm(total_feats, torch.transpose(total_feats, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(total_feats * total_feats, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        cos_sim_mat = dot_product_mat / len_mat

        sim_label = torch.zeros_like(cos_sim_mat)
        for i in range(num_class + 1):
            if i in total_label:
                index_ = torch.where(total_label == i)
                index_split = torch.split(index_[0], 1)
                for indx in index_split:
                    sim_label[indx, index_split] = 1
        cos_sim_mat_flatten = torch.flatten(cos_sim_mat, 0)
        sim_label_flatten = torch.flatten(sim_label, 0)
        sim_loss_value = F.hinge_embedding_loss(cos_sim_mat_flatten, sim_label_flatten, margin=0.1)
        return sim_loss_value

    def _bbox_forward_train(self, query_feats: List[Tensor],
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
        # 将subclass label转换为supclass label
        query_gt_sup_labels = self.conver_gt_cls(query_gt_labels)
        gt_query_supclass_cls = self.conver_cls(labels)
        gt_support_supclass_cls_temp = self.conver_cls(support_gt_labels)
        # 将subclass feat 转换为 supclass feat
        sup_cls_feats, gt_support_supclass_cls = self.bbox_head.get_supcls(support_feat, gt_support_supclass_cls_temp)

        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': [], 'loss_supcls_': [], 'sup_acc_': []}
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size

        bbox_results = None
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            random_index = np.random.choice(
                range(query_gt_labels[img_id].size(0)))
            random_query_label = query_gt_labels[img_id][random_index]
            for i in range(support_feat.size(0)):
                # Following the official code, each query image only sample
                # one support class for training. Also the official code
                # only use the first class in `query_gt_labels` as support
                # class, while this code use random one sampled from
                # `query_gt_labels` instead.
                if support_gt_labels[i] == random_query_label:
                    bbox_results = self._bbox_forward(
                        query_roi_feats[start:end],
                        support_feat[i].unsqueeze(0))
                    single_loss_bbox = self.bbox_head.loss(
                        bbox_results['cls_score'], bbox_results['bbox_pred'],
                        query_rois[start:end], labels[start:end],
                        label_weights[start:end], bbox_targets[start:end],
                        bbox_weights[start:end])
                    for key in single_loss_bbox.keys():
                        loss_bbox[key].append(single_loss_bbox[key])

        # 计算supclass损失
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            random_index = np.random.choice(
                range(query_gt_sup_labels[img_id].size(0)))
            random_query_label = query_gt_sup_labels[img_id][random_index]
            for i in range(sup_cls_feats.size(0)):
                if gt_support_supclass_cls[i] == random_query_label:

                    query_roi_feats_temp = query_roi_feats[start:end]
                    support_roi_feats = sup_cls_feats[i].unsqueeze(0)

                    roi_feats = self.aggregation_layer(query_feat=query_roi_feats_temp.unsqueeze(-1).unsqueeze(-1),
                                                       support_feat=support_roi_feats.view(1, -1, 1, 1))[0]
                    supcls_score = self.bbox_head.forward_supcls_(roi_feats.squeeze(-1).squeeze(-1))

                    loss_supcls = self.bbox_head.loss_supcls_fun_(supcls_score, gt_query_supclass_cls[start:end],
                                                                  torch.ones_like(gt_query_supclass_cls[start:end]))

                    for key in loss_supcls.keys():
                        loss_bbox[key].append(loss_supcls[key])

        if bbox_results is not None:
            for key in loss_bbox.keys():
                if 'acc' in key:
                    loss_bbox[key] = torch.cat(loss_bbox[key]).mean()
                else:
                    loss_bbox[key] = torch.stack(loss_bbox[key]).sum() / batch_size

        # meta classification loss
        if self.bbox_head.with_meta_cls_loss:
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        if self.bbox_head.with_supcls_loss:
            supcls_score = self.bbox_head.forward_supcls(sup_cls_feats)
            # supcls_labels = torch.cat(gt_support_supclass_cls)
            loss_supcls = self.bbox_head.loss_supcls_fun(
                supcls_score, gt_support_supclass_cls,
                torch.ones_like(gt_support_supclass_cls))
            loss_bbox.update(loss_supcls)

        sim_loss_value = self.sim_loss(support_feat, query_roi_feats, labels, support_gt_labels)
        loss_bbox['sim_loss'] = sim_loss_value * 0.2
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward_train_score(self, query_feats: List[Tensor],
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
        # 将subclass label转换为supclass label
        query_gt_sup_labels = self.conver_gt_cls(query_gt_labels)
        gt_query_supclass_cls = self.conver_cls(labels)
        gt_support_supclass_cls_temp = self.conver_cls(support_gt_labels)
        # 将subclass feat 转换为 supclass feat
        sup_cls_feats, gt_support_supclass_cls = self.bbox_head.get_supcls(support_feat, gt_support_supclass_cls_temp)

        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': [], 'loss_supcls_': [], 'sup_acc_': []}
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size
        bbox_results = None
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            random_index = np.random.choice(
                range(query_gt_labels[img_id].size(0)))
            random_query_label = query_gt_labels[img_id][random_index]
            sim_list = []
            sim_label_list = []
            for i in range(support_feat.size(0)):
                # Following the official code, each query image only sample
                # one support class for training. Also the official code
                # only use the first class in `query_gt_labels` as support
                # class, while this code use random one sampled from
                # `query_gt_labels` instead.
                if support_gt_labels[i] == random_query_label:
                    bbox_results = self._bbox_forward(
                        query_roi_feats[start:end],
                        support_feat[i].unsqueeze(0))
                    single_loss_bbox = self.bbox_head.loss(
                        bbox_results['cls_score'], bbox_results['bbox_pred'],
                        query_rois[start:end], labels[start:end],
                        label_weights[start:end], bbox_targets[start:end],
                        bbox_weights[start:end])
                    for key in single_loss_bbox.keys():
                        loss_bbox[key].append(single_loss_bbox[key])

                    sim_rate = torch.cosine_similarity(torch.mean(query_roi_feats[start:end], dim=0).view(1, -1),
                                                       support_feat[i].unsqueeze(0), 1)
                    sim_list.append(sim_rate)
                    sim_label_list.append(1)
                else:
                    sim_rate = torch.cosine_similarity(torch.mean(query_roi_feats[start:end], dim=0).view(1, -1),
                                                       support_feat[i].unsqueeze(0), 1)
                    sim_list.append(sim_rate)
                    sim_label_list.append(0)

            sim_list = torch.cat(sim_list, 0)
            sim_label_list = torch.tensor(sim_label_list).cuda()
            # pdb.set_trace()
            loss_sim_dict = self.bbox_head.loss_sim_fun(sim_list, sim_label_list, torch.ones_like(sim_list))

        # 计算supclass损失
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            random_index = np.random.choice(
                range(query_gt_sup_labels[img_id].size(0)))
            random_query_label = query_gt_sup_labels[img_id][random_index]
            for i in range(sup_cls_feats.size(0)):
                if gt_support_supclass_cls[i] == random_query_label:

                    query_roi_feats_temp = query_roi_feats[start:end]
                    support_roi_feats = sup_cls_feats[i].unsqueeze(0)

                    roi_feats = self.aggregation_layer(query_feat=query_roi_feats_temp.unsqueeze(-1).unsqueeze(-1),
                                                       support_feat=support_roi_feats.view(1, -1, 1, 1))[0]
                    supcls_score = self.bbox_head.forward_supcls_(roi_feats.squeeze(-1).squeeze(-1))

                    loss_supcls = self.bbox_head.loss_supcls_fun_(supcls_score, gt_query_supclass_cls[start:end],
                                                                  torch.ones_like(gt_query_supclass_cls[start:end]))

                    for key in loss_supcls.keys():
                        loss_bbox[key].append(loss_supcls[key])

        if bbox_results is not None:
            for key in loss_bbox.keys():
                if 'acc' in key:
                    loss_bbox[key] = torch.cat(loss_bbox[key]).mean()
                else:
                    loss_bbox[key] = torch.stack(loss_bbox[key]).sum() / batch_size

        # meta classification loss
        if self.bbox_head.with_meta_cls_loss:
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        if self.bbox_head.with_supcls_loss:
            supcls_score = self.bbox_head.forward_supcls(sup_cls_feats)
            # supcls_labels = torch.cat(gt_support_supclass_cls)
            loss_supcls = self.bbox_head.loss_supcls_fun(
                supcls_score, gt_support_supclass_cls,
                torch.ones_like(gt_support_supclass_cls))
            loss_bbox.update(loss_supcls)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward_train_cat_sup(self, query_feats: List[Tensor],
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

        # 将subclass label转换为supclass label
        query_gt_sup_labels = self.conver_gt_cls(query_gt_labels)
        gt_query_supclass_cls = self.conver_cls(labels)
        gt_support_supclass_cls_temp = self.conver_cls(support_gt_labels)
        # 将subclass feat 转换为 supclass feat

        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': [], 'loss_supcls_': [], 'sup_acc_': []}
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size
        bbox_results = None
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            cls_scores = []
            bbox_preds = []
            background = []
            if self.use_gnn:
                support_feat_, query_roi_feats_ = self.gnn._process_per_class(query_rois[start:end],
                                                                              query_roi_feats[start:end], support_feat)
            else:
                support_feat_, query_roi_feats_ = support_feat, query_roi_feats[start:end]

            # sim_rate = torch.cosine_similarity(support_feat[16].unsqueeze(0), query_roi_feats[start:end], 1)
            for i in range(support_feat_.size(0)):
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

        # 计算supclass损失
        sup_cls_feats, gt_support_supclass_cls = self.bbox_head.get_supcls(support_feat_, gt_support_supclass_cls_temp)
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            random_index = np.random.choice(
                range(query_gt_sup_labels[img_id].size(0)))
            random_query_label = query_gt_sup_labels[img_id][random_index]
            for i in range(sup_cls_feats.size(0)):
                if gt_support_supclass_cls[i] == random_query_label:

                    query_roi_feats_temp = query_roi_feats[start:end]
                    support_roi_feats = sup_cls_feats[i].unsqueeze(0)

                    roi_feats = self.aggregation_layer(query_feat=query_roi_feats_temp.unsqueeze(-1).unsqueeze(-1),
                                                       support_feat=support_roi_feats.view(1, -1, 1, 1))[0]
                    supcls_score = self.bbox_head.forward_supcls_(roi_feats.squeeze(-1).squeeze(-1))

                    loss_supcls = self.bbox_head.loss_supcls_fun_(supcls_score, gt_query_supclass_cls[start:end],
                                                                  torch.ones_like(gt_query_supclass_cls[start:end]))

                    for key in loss_supcls.keys():
                        loss_bbox[key].append(loss_supcls[key])

        if bbox_results is not None:
            for key in loss_bbox.keys():
                if 'acc' in key:
                    loss_bbox[key] = torch.cat(loss_bbox[key]).mean()
                else:
                    loss_bbox[key] = torch.stack(loss_bbox[key]).sum() / batch_size

        # meta classification loss
        if self.bbox_head.with_meta_cls_loss:
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat_)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        if self.use_meta_sup:
            if self.bbox_head.with_supcls_loss:
                supcls_score = self.bbox_head.forward_supcls(sup_cls_feats)
                # supcls_labels = torch.cat(gt_support_supclass_cls)
                loss_supcls = self.bbox_head.loss_supcls_fun(
                    supcls_score, gt_support_supclass_cls,
                    torch.ones_like(gt_support_supclass_cls))
                loss_bbox.update(loss_supcls)
                # sim_loss_value = self.sim_loss(support_feat, query_roi_feats, labels, support_gt_labels)
        # loss_bbox['sim_loss'] = sim_loss_value
        # pdb.set_trace()
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward_train_2cat_sup(self, query_feats: List[Tensor],
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

        # 将subclass label转换为supclass label
        query_gt_sup_labels = self.conver_gt_cls(query_gt_labels)
        gt_query_supclass_cls = self.conver_cls(labels)

        gt_support_supclass_cls_temp = self.conver_cls(support_gt_labels)
        # 将subclass feat 转换为 supclass feat

        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': [], 'loss_supcls_': [], 'sup_acc_': []}
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size
        bbox_results = None
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            cls_scores = []
            bbox_preds = []
            background = []
            if self.use_gnn:
                support_feat_, query_roi_feats_ = self.gnn._process_per_class(query_rois[start:end],
                                                                              query_roi_feats[start:end], support_feat)
            else:
                support_feat_, query_roi_feats_ = support_feat, query_roi_feats[start:end]

            # sim_rate = torch.cosine_similarity(support_feat[16].unsqueeze(0), query_roi_feats[start:end], 1)
            for i in range(support_feat_.size(0)):
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

        # 计算supclass损失
        sup_cls_feats, gt_support_supclass_cls = self.bbox_head.get_supcls(support_feat, gt_support_supclass_cls_temp)
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            sup_cls_scores = []
            sup_background = []
            for i in range(sup_cls_feats.size(0)):
                query_roi_feats_temp = query_roi_feats[start:end]
                support_roi_feats = sup_cls_feats[i].unsqueeze(0)

                roi_feats = self.aggregation_layer(query_feat=query_roi_feats_temp.unsqueeze(-1).unsqueeze(-1),
                                                   support_feat=support_roi_feats.view(1, -1, 1, 1))[0]
                supcls_score = self.bbox_head.forward_supcls_(roi_feats.squeeze(-1).squeeze(-1))

                sup_cls_scores.append(supcls_score[:, i:i + 1])
                sup_background.append(supcls_score[:, -1:])
            sup_background = torch.mean(torch.cat(sup_background, 1), dim=1).view(sup_background[0].shape[0], -1)
            sup_cls_scores.append(sup_background)
            sup_cls_scores = torch.cat(sup_cls_scores, 1)
            loss_supcls = self.bbox_head.loss_supcls_fun_(sup_cls_scores, gt_query_supclass_cls[start:end],
                                                          torch.ones_like(gt_query_supclass_cls[start:end]))
            for key in loss_supcls.keys():
                loss_bbox[key].append(loss_supcls[key])

        if bbox_results is not None:
            for key in loss_bbox.keys():
                if 'acc' in key:
                    loss_bbox[key] = torch.cat(loss_bbox[key]).mean()
                else:
                    loss_bbox[key] = torch.stack(loss_bbox[key]).sum() / batch_size

        # meta classification loss
        if self.bbox_head.with_meta_cls_loss:
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat_)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        if self.use_meta_sup:
            if self.bbox_head.with_supcls_loss:
                supcls_score = self.bbox_head.forward_supcls(sup_cls_feats)
                # supcls_labels = torch.cat(gt_support_supclass_cls)
                loss_supcls = self.bbox_head.loss_supcls_fun(
                    supcls_score, gt_support_supclass_cls,
                    torch.ones_like(gt_support_supclass_cls))
                loss_bbox.update(loss_supcls)
                # sim_loss_value = self.sim_loss(support_feat, query_roi_feats, labels, support_gt_labels)
        # loss_bbox['sim_loss'] = sim_loss_value
        # pdb.set_trace()
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward_train_2cat_sup_asy(self, query_feats: List[Tensor],
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

        # 将subclass label转换为supclass label
        query_gt_sup_labels = self.conver_gt_cls(query_gt_labels)
        gt_query_supclass_cls_ = self.conver_cls_asy(labels)
        gt_query_supclass_cls = gt_query_supclass_cls_ + self.bbox_head.num_classes - 7
        supcls_index = self.bbox_head.num_classes - 7

        gt_support_supclass_cls_temp = self.conver_cls(support_gt_labels)
        # 将subclass feat 转换为 supclass feat

        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': []}
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size
        bbox_results = None
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            cls_scores = []
            bbox_preds = []
            background = []
            if self.use_gnn:
                support_feat_, query_roi_feats_ = self.gnn._process_per_class(query_rois[start:end],
                                                                              query_roi_feats[start:end], support_feat)
            else:
                support_feat_, query_roi_feats_ = support_feat, query_roi_feats[start:end]

            # sim_rate = torch.cosine_similarity(support_feat[16].unsqueeze(0), query_roi_feats[start:end], 1)
            for i in range(support_feat_.size(0)):
                bbox_results = self._bbox_forward(
                    query_roi_feats_,
                    support_feat_[i].unsqueeze(0))

                # bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)
                cls_scores.append(bbox_results['cls_score'][:, i:i + 1])
                bbox_preds.append(bbox_results['bbox_pred'][:, i * 4:(i + 1) * 4])
                background.append(bbox_results['cls_score'][:, supcls_index:].unsqueeze(0))

            background = torch.mean(torch.cat(background, 0), dim=0)
            cls_scores.append(background)
            cls_scores = torch.cat(cls_scores, 1)
            bbox_preds = torch.cat(bbox_preds, 1)

            input_labels = torch.cat([labels[start:end].view(-1, 1), gt_query_supclass_cls[start:end].view(-1, 1)], 1)

            single_loss_bbox = self.bbox_head.loss_asy(
                cls_scores, bbox_preds,
                query_rois[start:end], input_labels,
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
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat_)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        if self.use_meta_sup:
            if self.bbox_head.with_supcls_loss:
                supcls_score = self.bbox_head.forward_supcls(sup_cls_feats)
                # supcls_labels = torch.cat(gt_support_supclass_cls)
                loss_supcls = self.bbox_head.loss_supcls_fun(
                    supcls_score, gt_support_supclass_cls,
                    torch.ones_like(gt_support_supclass_cls))
                loss_bbox.update(loss_supcls)
                # sim_loss_value = self.sim_loss(support_feat, query_roi_feats, labels, support_gt_labels)
        # loss_bbox['sim_loss'] = sim_loss_value
        # pdb.set_trace()
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward_train_cat_norsup(self, query_feats: List[Tensor],
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

        # 将subclass label转换为supclass label
        query_gt_sup_labels = self.conver_gt_cls(query_gt_labels)
        gt_query_supclass_cls = self.conver_cls(labels)
        gt_support_supclass_cls_temp = self.conver_cls(support_gt_labels)
        # 将subclass feat 转换为 supclass feat

        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': [], 'loss_supcls_': [], 'sup_acc_': []}
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size
        bbox_results = None
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            cls_scores = []
            bbox_preds = []
            background = []
            if self.use_gnn:
                support_feat_, query_roi_feats_ = self.gnn._process_per_class(query_rois[start:end],
                                                                              query_roi_feats[start:end], support_feat)
            else:
                support_feat_, query_roi_feats_ = support_feat, query_roi_feats[start:end]

            # sim_rate = torch.cosine_similarity(support_feat[16].unsqueeze(0), query_roi_feats[start:end], 1)
            for i in range(support_feat_.size(0)):
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

        # 计算supclass损失
        sup_cls_feats, gt_support_supclass_cls = self.bbox_head.get_supcls(support_feat_, gt_support_supclass_cls_temp)
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge

            query_roi_feats_temp = query_roi_feats[start:end]
            supcls_score = self.bbox_head.forward_supcls_(query_roi_feats_temp.squeeze(-1).squeeze(-1))
            loss_supcls = self.bbox_head.loss_supcls_fun_(supcls_score, gt_query_supclass_cls[start:end],
                                                          torch.ones_like(gt_query_supclass_cls[start:end]))
            for key in loss_supcls.keys():
                loss_bbox[key].append(loss_supcls[key])

        if bbox_results is not None:
            for key in loss_bbox.keys():
                if 'acc' in key:
                    loss_bbox[key] = torch.cat(loss_bbox[key]).mean()
                else:
                    loss_bbox[key] = torch.stack(loss_bbox[key]).sum() / batch_size

        # meta classification loss
        if self.bbox_head.with_meta_cls_loss:
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat_)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

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
        if self.use_text:
            sen_batches = self.text_aug.get_sentence_batches(support_gt_labels)
            support_feat = self.text_aug.forward(sen_batches, support_feat, fusion=True)
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
            if self.use_gnn:
                support_feat_, query_roi_feats_ = self.gnn._process_per_class(query_rois[start:end],
                                                                              query_roi_feats[start:end], support_feat)
            else:
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
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        if self.with_sim_loss:
            sim_loss_value = self.cal_loss()
            loss_bbox['sim_loss'] = sim_loss_value
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward_train_cat_text(self, query_feats: List[Tensor],
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

        if self.use_text:
            sen_batches = self.text_aug.get_sentence_batches(support_gt_labels)
            support_feat = self.text_aug.forward(sen_batches, support_feat, fusion=True)
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
            if self.use_gnn:
                support_feat_, query_roi_feats_ = self.gnn._process_per_class(query_rois[start:end],
                                                                              query_roi_feats[start:end], support_feat)
            else:
                support_feat_, query_roi_feats_ = support_feat, query_roi_feats[start:end]

            # sim_rate = torch.cosine_similarity(support_feat[16].unsqueeze(0), query_roi_feats[start:end], 1)
            for i in range(support_feat_.size(0)):
                # pdb.set_trace()
                if self.with_text_sim_aug:
                    bbox_results = self._bbox_forward(
                        query_roi_feats_,
                        support_feat_[i].unsqueeze(0), text_feat=embeddings[i].unsqueeze(0))
                else:
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
            if self.use_gnn:
                support_feat_, query_roi_feats_ = self.gnn._process_per_class(query_rois[start:end],
                                                                              query_roi_feats[start:end], support_feat)
            else:
                support_feat_, query_roi_feats_ = support_feat, query_roi_feats[start:end]
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat_)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        if self.with_sim_loss:
            sim_loss_value = self.cal_loss()
            loss_bbox['sim_loss'] = sim_loss_value
        bbox_results.update(loss_bbox=loss_bbox)
        # pdb.set_trace()
        if self.with_text_sim_loss:
            text_sim_loss_value = self.cal_text_sin_loss(embeddings, support_feat)
            loss_bbox['text_sim_loss'] = text_sim_loss_value
        bbox_results.update(loss_bbox=loss_bbox)

        if hasattr(self.shared_head, 'loss_kd'):
            loss_bbox.update(self.shared_head.loss_kd)
            bbox_results.update(loss_bbox=loss_bbox)

        if self.text_aug.constrain:
            loss_bbox['c_loss'] = self.text_aug.constrain_loss['c_loss']
            bbox_results.update(loss_bbox=loss_bbox)

        return bbox_results

    def _bbox_forward_train_text(self, query_feats: List[Tensor],
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

        if self.use_text:
            sen_batches = self.text_aug.get_sentence_batches(support_gt_labels)
            support_feat = self.text_aug.forward(sen_batches, support_feat, fusion=True)
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
            random_index = np.random.choice(
                range(query_gt_labels[img_id].size(0)))
            random_query_label = query_gt_labels[img_id][random_index]
            for i in range(support_feat.size(0)):
                # Following the official code, each query image only sample
                # one support class for training. Also the official code
                # only use the first class in `query_gt_labels` as support
                # class, while this code use random one sampled from
                # `query_gt_labels` instead.
                if support_gt_labels[i] == random_query_label:
                    bbox_results = self._bbox_forward(
                        query_roi_feats[start:end],
                        support_feat[i].unsqueeze(0))
                    single_loss_bbox = self.bbox_head.loss(
                        bbox_results['cls_score'], bbox_results['bbox_pred'],
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
            if self.use_gnn:
                support_feat_, query_roi_feats_ = self.gnn._process_per_class(query_rois[start:end],
                                                                              query_roi_feats[start:end], support_feat)
            else:
                support_feat_, query_roi_feats_ = support_feat, query_roi_feats[start:end]
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat_)
            meta_cls_labels = torch.cat(support_gt_labels)
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_bbox.update(loss_meta_cls)

        if self.with_sim_loss:
            sim_loss_value = self.cal_loss()
            loss_bbox['sim_loss'] = sim_loss_value
        bbox_results.update(loss_bbox=loss_bbox)
        # pdb.set_trace()
        if self.with_text_sim_loss:
            text_sim_loss_value = self.cal_text_sin_loss(embeddings, support_feat)
            loss_bbox['text_sim_loss'] = text_sim_loss_value
        bbox_results.update(loss_bbox=loss_bbox)

        if hasattr(self.shared_head, 'loss_kd'):
            loss_bbox.update(self.shared_head.loss_kd)
            bbox_results.update(loss_bbox=loss_bbox)

        if self.text_aug.constrain:
            loss_bbox['c_loss'] = self.text_aug.constrain_loss['c_loss']
            bbox_results.update(loss_bbox=loss_bbox)

        return bbox_results

    def cal_text_sin_loss(self, embeddings, support_feat):
        avg_for_cos_sim = 0
        fore_num = embeddings.shape[0]
        cos_sim = torch.cosine_similarity(embeddings, support_feat, 1)
        for i in range(fore_num):
            avg_for_cos_sim = cos_sim[i] + avg_for_cos_sim
        avg_for_cos_sim = avg_for_cos_sim / fore_num
        cal_loss = 2 - avg_for_cos_sim
        return cal_loss * 0.1

    def cal_loss(self):
        weight_p = self.bbox_head.fc_cls.weight
        weight_c = self.bbox_head.fc_meta.weight
        # pdb.set_trace()
        fore_num = weight_p.shape[0] - 1
        avg_for_cos_sim = 0

        cos_sim = torch.cosine_similarity(weight_p[:fore_num], weight_c, 1)
        for i in range(fore_num):
            avg_for_cos_sim = cos_sim[i] + avg_for_cos_sim
        avg_for_cos_sim = avg_for_cos_sim / fore_num
        cal_loss = 2 - avg_for_cos_sim + cos_sim[-1]
        return cal_loss * 0.5

    def extract_query_roi_feat(self, feats: List[Tensor],
                               rois: Tensor) -> Tensor:
        """Extracting query BBOX features, which is used in both training and
        testing.

        Args:
            feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            rois (Tensor): shape with (m, 5).

        Returns:
            Tensor: RoI features with shape (N, C).
        """
        roi_feats = self.bbox_roi_extractor(
            feats[:self.bbox_roi_extractor.num_inputs], rois)

        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        return roi_feats

    def extract_support_feats(self, feats: List[Tensor]) -> List[Tensor]:
        """Forward support features through shared layers.

        Args:
            feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).

        Returns:
            list[Tensor]: List of support features, each item
                with shape (N, C).
        """
        out = []
        if self.with_shared_head:
            for lvl in range(len(feats)):
                out.append(self.shared_head.forward_support(feats[lvl]))
        else:
            out = feats
        return out

    def _bbox_forward(self, query_roi_feats,
                      support_roi_feats, text_feat=None):
        """Box head forward function used in both training and testing.

        Args:
            query_roi_feats (Tensor): Query roi features with shape (N, C).
            support_roi_feats (Tensor): Support features with shape (1, C).

        Returns:
             dict: A dictionary of predicted results.
        """
        # roi_feats = self.aggregation_layer(
        #     query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
        #     support_feat=support_roi_feats.view(1, -1, 1, 1))[0]
        # if text_feat is not None:
        #     roi_feats = self.aggregation_layer(query_feat=roi_feats, support_feat=text_feat.view(1, -1, 1, 1))[0]
        roi_feats = query_roi_feats.unsqueeze(-1).unsqueeze(-1)
        cls_score, bbox_pred = self.bbox_head(
            roi_feats.squeeze(-1).squeeze(-1))

        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    # def _bbox_forward(self, query_roi_feats: Tensor,
    #                   support_roi_feats: Tensor) -> Dict:
    #     """Box head forward function used in both training and testing.
    #
    #     Args:
    #         query_roi_feats (Tensor): Roi features with shape (N, C).
    #         support_roi_feats (Tensor): Roi features with shape (1, C).
    #
    #     Returns:
    #          dict: A dictionary of predicted results.
    #     """
    #     # feature aggregation
    #     roi_feats = self.aggregation_layer(
    #         query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
    #         support_feat=support_roi_feats.view(1, -1, 1, 1))
    #     roi_feats = torch.cat(roi_feats, dim=1)
    #     roi_feats = torch.cat((roi_feats, query_roi_feats), dim=1)
    #     cls_score, bbox_pred = self.bbox_head(roi_feats)
    #     bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
    #     return bbox_results

    def conver_cls(self, labels):
        gt_supclass_cls = []
        if isinstance(labels, list):
            for k in range(len(labels)):
                label = labels[k].item()
                for i, cls in enumerate(self.classes):
                    if label == i:
                        cls = cls.replace(' ', '_')
                        reason_id = self.reason_cls[cls]
                        gt_supclass_cls.append(reason_id[0] - 100)
                if label == len(self.classes):
                    gt_supclass_cls.append(6)
        else:
            for k in range(labels.shape[0]):
                label = labels[k].item()
                for i, cls in enumerate(self.classes):
                    if label == i:
                        cls = cls.replace(' ', '_')
                        reason_id = self.reason_cls[cls]
                        gt_supclass_cls.append(reason_id[0] - 100)
                if label == len(self.classes):
                    gt_supclass_cls.append(6)

        return torch.tensor(gt_supclass_cls).cuda()

    def conver_cls_asy(self, labels):
        gt_supclass_cls = []
        if isinstance(labels, list):
            for k in range(len(labels)):
                label = labels[k].item()
                for i, cls in enumerate(self.classes):
                    if label == i:
                        cls = cls.replace(' ', '_')
                        reason_id = self.reason_cls[cls]
                        gt_supclass_cls.append(reason_id[0] - 100)
                if label == (len(self.classes) + 7):
                    gt_supclass_cls.append(6)
        else:
            for k in range(labels.shape[0]):
                label = labels[k].item()
                for i, cls in enumerate(self.classes):
                    if label == i:
                        cls = cls.replace(' ', '_')
                        reason_id = self.reason_cls[cls]
                        gt_supclass_cls.append(reason_id[0] - 100)
                if label == (len(self.classes) + 7):
                    gt_supclass_cls.append(6)

        return torch.tensor(gt_supclass_cls).cuda()

    def conver_gt_cls(self, labels):
        gt_supclass_cls = []
        if isinstance(labels, list):
            for classes in labels:
                temp = torch.zeros_like(classes)
                for k in range(classes.shape[0]):
                    label = classes[k].item()
                    for i, cls in enumerate(self.classes):
                        if label == i:
                            cls = cls.replace(' ', '_')
                            reason_id = self.reason_cls[cls]
                            temp[k] = reason_id[0] - 100
                gt_supclass_cls.append(temp)
        return gt_supclass_cls

    def simple_test(self,
                    query_feats: List[Tensor],
                    support_feats_dict: Dict,
                    proposal_list: List[Tensor],
                    query_img_metas: List[Dict],
                    gt_labels: List[Tensor],
                    rescale: bool = False) -> List[List[np.ndarray]]:
        """Test without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            proposal_list (list[Tensors]): list of region proposals.
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            rescale (bool): Whether to rescale the results. Default: False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        flag = self.flag
        if flag == 3 or flag == 4:  # 3 4  cat + sup cls| 2cat+sup cls
            det_bboxes, det_labels = self.simple_test_sup_exp(
                query_feats,
                support_feats_dict,
                query_img_metas,
                gt_labels,
                proposal_list,
                self.test_cfg,
                rescale=rescale)
        elif flag == 5:  # 5 cat + normal sup cls
            det_bboxes, det_labels = self.simple_test_norsup(
                query_feats,
                support_feats_dict,
                query_img_metas,
                gt_labels,
                proposal_list,
                self.test_cfg,
                rescale=rescale)
        elif flag == 7:
            det_bboxes, det_labels = self.simple_test_bboxes(
                query_feats,
                support_feats_dict,
                query_img_metas,
                gt_labels,
                proposal_list,
                self.test_cfg,
                rescale=rescale)
        elif flag == 8 or flag == 9:
            det_bboxes, det_labels = self.simple_test_bboxes_text(
                query_feats,
                support_feats_dict,
                query_img_metas,
                gt_labels,
                proposal_list,
                self.test_cfg,
                rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

    def get_subclass(self, supclass):
        subclass = []
        for i, cls in enumerate(self.classes):
            cls = cls.replace(' ', '_')
            reason_id = self.reason_cls[cls]
            if reason_id[0] - 100 in supclass:
                subclass.append(i)
        return subclass

    # def fusion_cls(self, supcls_score, cls_score, support_supclass_cls):
    #     supcls_score = F.softmax(supcls_score, dim=-1) if supcls_score is not None else None
    #     mask = torch.ones_like(cls_score)
    #     # pdb.set_trace()
    #     for i in range(supcls_score.shape[0]):
    #         cls = i
    #         index_ = torch.where(support_supclass_cls == cls)
    #         if index_[0].shape[0] == 0:
    #             continue
    #         index_split = torch.split(index_[0], 1)
    #         mask[:, index_split] = mask[:, index_split] * supcls_score[:, i:i + 1]
    #     # mask[:, -1:] = mask[:, -1:] * supcls_score[:, -1:]
    #     cls_score = cls_score * mask
    #     return cls_score

    def fusion_cls(self, supcls_score, cls_score, support_supclass_cls):
        supcls_score = F.softmax(supcls_score, dim=-1) if supcls_score is not None else None
        cls_score = F.softmax(cls_score, dim=-1) if supcls_score is not None else None
        mask = torch.zeros_like(cls_score)
        # pdb.set_trace()
        for i in range(supcls_score.shape[0]):
            cls = i
            index_ = torch.where(support_supclass_cls == cls)
            if index_[0].shape[0] == 0:
                continue
            index_split = torch.split(index_[0], 1)
            mask[:, index_split] = supcls_score[:, i:i + 1]
        # mask[:, -1:] = mask[:, -1:] * supcls_score[:, -1:]
        cls_score = 0.8 * cls_score + 0.2 * mask
        return cls_score

    def split_supcls_feats(self, feats, supcls_label, supcls_label_unique):
        supcls_feat_dict = {}
        index_dict = {}
        supcls_subcls_dict = {}
        for supcls in supcls_label_unique:
            index_ = torch.where(supcls_label == supcls)
            index_split = torch.split(index_[0], 1)
            supcls_feat_dict[supcls.item()] = feats[index_split, :]
            index_dict[supcls.item()] = index_split

        for i, cls in enumerate(self.classes):
            cls = cls.replace(' ', '_')
            reason_id = self.reason_cls[cls]
            if reason_id[0] - 100 in supcls_label_unique:
                if reason_id[0] - 100 not in supcls_subcls_dict.keys():
                    supcls_subcls_dict[reason_id[0] - 100] = []
                supcls_subcls_dict[reason_id[0] - 100].append(i)

        return supcls_feat_dict, index_dict, supcls_subcls_dict

    def simple_test_bboxes_instance(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            gt_labels: List[Tensor],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        # gt_supclass_cls = self.conver_cls(gt_labels[0][0])
        # subclass = self.get_subclass(gt_supclass_cls)
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)

        rois = bbox2roi(proposals)

        support_key = []
        support_feats = []
        for key in support_feats_dict.keys():
            support_feats.append(support_feats_dict[key])
            support_key.append(key)

        support_feats = torch.cat(support_feats, 0)
        support_key = torch.tensor(support_key)
        support_supclass_cls = self.conver_cls(support_key)
        sup_cls_feats, support_supclass_cls_temp = self.bbox_head.get_supcls(support_feats, support_supclass_cls)
        support_supcls_feats_dict = dict()
        supcls_scores_dict = {}
        for i in range(support_supclass_cls_temp.shape[0]):
            key = support_supclass_cls_temp[i].item()
            support_supcls_feats_dict[key] = sup_cls_feats[i, :]

        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)
        for class_id in support_supcls_feats_dict.keys():
            support_feat = support_supcls_feats_dict[class_id]
            roi_feats = \
                self.aggregation_layer(query_roi_feats.unsqueeze(-1).unsqueeze(-1), support_feat.view(1, -1, 1, 1))[0]
            supcls_score = self.bbox_head.forward_supcls_(roi_feats.squeeze(-1).squeeze(-1))
            supcls_score = F.softmax(supcls_score, dim=-1)
            supcls_scores_dict[class_id] = supcls_score[:, class_id:class_id + 1]

        supcls_scores = [
            supcls_scores_dict[i] if i in supcls_scores_dict.keys() else
            torch.zeros_like(supcls_scores_dict[list(supcls_scores_dict.keys())[0]])
            for i in range(6)
        ]
        supcls_scores = torch.cat(supcls_scores, dim=1)
        supcls_scores = F.softmax(supcls_scores, dim=-1) if supcls_scores is not None else None
        value, index_ = torch.max(supcls_scores, 1)

        gt_supclass_cls = []
        for c in index_:
            cl = c.item()
            if cl in gt_supclass_cls:
                continue
            gt_supclass_cls.append(cl)
        gt_supclass_cls = torch.tensor(gt_supclass_cls).cuda()
        # subclass = self.get_subclass(gt_supclass_cls)
        # pdb.set_trace()
        supcls_feat_dict, index_dict, supcls_subcls_dict = self.split_supcls_feats(query_roi_feats, index_,
                                                                                   gt_supclass_cls)

        cls_scores_dict, bbox_preds_dict = {}, {}
        num_classes = self.bbox_head.num_classes
        for supcls_id in supcls_feat_dict.keys():
            subclass = supcls_subcls_dict[supcls_id]
            feats = supcls_feat_dict[supcls_id]
            index = index_dict[supcls_id]

            for class_id in support_feats_dict.keys():
                # pdb.set_trace()
                if class_id not in subclass:
                    continue
                support_feat = support_feats_dict[class_id]
                bbox_results = self._bbox_forward(feats, support_feat)
                cls_scores_temp = torch.zeros(query_roi_feats.shape[0], 1).cuda()
                bbox_preds_temp = torch.zeros(query_roi_feats.shape[0], 4).cuda()
                bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)
                cls_scores_temp[index, :] = bbox_results['cls_score'][:, class_id:class_id + 1]
                bbox_preds_temp[index, :] = bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]
                cls_scores_dict[class_id] = cls_scores_temp
                bbox_preds_dict[class_id] = bbox_preds_temp

        for class_id in support_feats_dict.keys():
            support_feat = support_feats_dict[class_id]
            bbox_results = self._bbox_forward(query_roi_feats, support_feat)
            bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)
            if cls_scores_dict.get(num_classes, None) is None:
                cls_scores_dict[num_classes] = \
                    bbox_results['cls_score'][:, -1:]
            else:
                cls_scores_dict[num_classes] += \
                    bbox_results['cls_score'][:, -1:]

        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())
        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
            for i in range(num_classes + 1)
        ]
        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
            for i in range(num_classes)
        ]
        cls_score = torch.cat(cls_scores, dim=1)
        bbox_pred = torch.cat(bbox_preds, dim=1)

        # cls_score = self.fusion_cls(supcls_scores, cls_score, support_supclass_cls)
        # cls_score = cls_score + cls_score_
        # split batch bbox prediction back to each image
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels

    def simple_test_sup(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            gt_labels: List[Tensor],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        softmax_flag = False
        # gt_supclass_cls = self.conver_cls(gt_labels[0][0])
        # subclass = self.get_subclass(gt_supclass_cls)
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)

        rois = bbox2roi(proposals)
        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)

        support_key = []
        support_feats = []
        for key in support_feats_dict.keys():
            support_feats.append(support_feats_dict[key])
            support_key.append(key)

        support_feats = torch.cat(support_feats, 0)
        if self.use_gnn:
            support_feats, query_roi_feats = self.gnn._process_per_class(rois, query_roi_feats, support_feats)

        support_key = torch.tensor(support_key)
        support_supclass_cls = self.conver_cls(support_key)
        sup_cls_feats, support_supclass_cls_temp = self.bbox_head.get_supcls(support_feats, support_supclass_cls)
        support_supcls_feats_dict = dict()
        supcls_scores_dict = {}
        for i in range(support_supclass_cls_temp.shape[0]):
            key = support_supclass_cls_temp[i].item()
            support_supcls_feats_dict[key] = sup_cls_feats[i, :]

        for i in range(support_feats.shape[0]):
            support_feats_dict[i] = support_feats[i, :].view(1, support_feats.shape[-1])

        for class_id in support_supcls_feats_dict.keys():
            support_feat = support_supcls_feats_dict[class_id]
            roi_feats = \
                self.aggregation_layer(query_roi_feats.unsqueeze(-1).unsqueeze(-1), support_feat.view(1, -1, 1, 1))[0]
            supcls_score = self.bbox_head.forward_supcls_(roi_feats.squeeze(-1).squeeze(-1))
            if softmax_flag:
                supcls_score = F.softmax(supcls_score, dim=-1)

            supcls_scores_dict[class_id] = supcls_score[:, class_id:class_id + 1]

        supcls_scores = [
            supcls_scores_dict[i] if i in supcls_scores_dict.keys() else
            torch.zeros_like(supcls_scores_dict[list(supcls_scores_dict.keys())[0]])
            for i in range(6)
        ]
        supcls_scores = torch.cat(supcls_scores, dim=1)
        supcls_scores = F.softmax(supcls_scores, dim=-1) if supcls_scores is not None else None
        value, index_ = torch.max(supcls_scores, 1)

        gt_supclass_cls = []
        for c in index_:
            cl = c.item()
            if cl in gt_supclass_cls:
                continue
            gt_supclass_cls.append(cl)
        gt_supclass_cls = torch.tensor(gt_supclass_cls).cuda()
        subclass = self.get_subclass(gt_supclass_cls)
        # # pdb.set_trace()

        cls_scores_dict, bbox_preds_dict = {}, {}
        num_classes = self.bbox_head.num_classes
        for class_id in support_feats_dict.keys():
            # pdb.set_trace()
            if class_id not in subclass:
                continue
            support_feat = support_feats_dict[class_id]
            bbox_results = self._bbox_forward(query_roi_feats, support_feat)
            if softmax_flag:
                bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)
            cls_scores_dict[class_id] = \
                bbox_results['cls_score'][:, class_id:class_id + 1]
            bbox_preds_dict[class_id] = \
                bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]

        for class_id in support_feats_dict.keys():

            support_feat = support_feats_dict[class_id]

            # sim_rate = torch.cosine_similarity(support_feats_dict[4], query_roi_feats, 1)
            bbox_results = self._bbox_forward(query_roi_feats, support_feat)

            if softmax_flag:
                bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)

            if cls_scores_dict.get(num_classes, None) is None:
                cls_scores_dict[num_classes] = \
                    bbox_results['cls_score'][:, -1:]
            else:
                cls_scores_dict[num_classes] += \
                    bbox_results['cls_score'][:, -1:]

        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())

        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
            for i in range(num_classes + 1)
        ]
        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
            for i in range(num_classes)
        ]
        cls_score = torch.cat(cls_scores, dim=1)
        bbox_pred = torch.cat(bbox_preds, dim=1)
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels

    def tool(self, supclass):
        list1 = supclass.cpu().numpy().tolist()
        sta_dict = dict()
        for ele in list1:
            if ele not in sta_dict.keys():
                sta_dict[ele] = 0

            sta_dict[ele] = sta_dict[ele] + 1
        return sta_dict

    def simple_test_sup_exp(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            gt_labels: List[Tensor],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """

        softmax_flag = False
        # gt_supclass_cls_ = self.conver_cls(gt_labels[0][0])
        # gt_subclass = self.get_subclass(gt_supclass_cls_)
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)

        rois = bbox2roi(proposals)
        query_roi_feats_ = self.extract_query_roi_feat(query_feats, rois)

        support_key = []
        support_feats = []
        for key in support_feats_dict.keys():
            support_feats.append(support_feats_dict[key])
            support_key.append(key)

        support_feats_ = torch.cat(support_feats, 0)
        if self.use_gnn:
            support_feats, query_roi_feats = self.gnn._process_per_class(rois, query_roi_feats_, support_feats_)

        support_key = torch.tensor(support_key)
        support_supclass_cls = self.conver_cls(support_key)

        sup_cls_feats, support_supclass_cls_temp = self.bbox_head.get_supcls(support_feats_, support_supclass_cls)

        support_supcls_feats_dict = dict()
        supcls_scores_dict = {}
        for i in range(support_supclass_cls_temp.shape[0]):
            key = support_supclass_cls_temp[i].item()
            support_supcls_feats_dict[key] = sup_cls_feats[i, :]

        for i in range(support_feats.shape[0]):
            support_feats_dict[i] = support_feats[i, :].view(1, support_feats.shape[-1])

        for class_id in support_supcls_feats_dict.keys():
            support_feat = support_supcls_feats_dict[class_id]
            roi_feats = \
                self.aggregation_layer(query_roi_feats_.unsqueeze(-1).unsqueeze(-1), support_feat.view(1, -1, 1, 1))[0]
            supcls_score = self.bbox_head.forward_supcls_(roi_feats.squeeze(-1).squeeze(-1))
            # print(supcls_score)
            # print(torch.mean(supcls_score, dim=1)[0])
            # print(torch.std(supcls_score, dim=1)[0])

            if softmax_flag:
                supcls_score = F.softmax(supcls_score, dim=-1)

            supcls_scores_dict[class_id] = supcls_score[:, class_id:class_id + 1]

        supcls_scores = [
            supcls_scores_dict[i] if i in supcls_scores_dict.keys() else
            torch.zeros_like(supcls_scores_dict[list(supcls_scores_dict.keys())[0]])
            for i in range(6)
        ]
        supcls_scores = torch.cat(supcls_scores, dim=1)
        supcls_scores = F.softmax(supcls_scores, dim=-1) if supcls_scores is not None else None
        value, index_ = torch.max(supcls_scores, 1)

        gt_supclass_cls = []
        for c in index_:
            cl = c.item()
            if cl in gt_supclass_cls:
                continue
            gt_supclass_cls.append(cl)
        gt_supclass_cls = torch.tensor(gt_supclass_cls).cuda()
        # print('gt:', gt_supclass_cls_)
        # print('pr:', gt_supclass_cls)
        # pdb.set_trace()
        subclass = self.get_subclass(gt_supclass_cls)
        # subclass = gt_subclass
        # pdb.set_trace()
        cls_scores_dict, bbox_preds_dict = {}, {}
        num_classes = self.bbox_head.num_classes
        for class_id in support_feats_dict.keys():
            # pdb.set_trace()
            if class_id not in subclass:
                continue
            support_feat = support_feats_dict[class_id]
            bbox_results = self._bbox_forward(query_roi_feats, support_feat)
            if softmax_flag:
                bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)
            cls_scores_dict[class_id] = \
                bbox_results['cls_score'][:, class_id:class_id + 1]
            # bbox_preds_dict[class_id] = \
            #     bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]

        for class_id in support_feats_dict.keys():

            support_feat = support_feats_dict[class_id]

            # sim_rate = torch.cosine_similarity(support_feats_dict[4], query_roi_feats, 1)
            bbox_results = self._bbox_forward(query_roi_feats, support_feat)

            bbox_preds_dict[class_id] = \
                bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]

            if softmax_flag:
                bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)

            if cls_scores_dict.get(num_classes, None) is None:
                cls_scores_dict[num_classes] = \
                    bbox_results['cls_score'][:, -1:]
            else:
                cls_scores_dict[num_classes] += \
                    bbox_results['cls_score'][:, -1:]

        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())
        # cls_scores_dict[num_classes] = (cls_scores_dict[num_classes] + supcls_scores_dict[7])/2

        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
            for i in range(num_classes + 1)
        ]

        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
            for i in range(num_classes)
        ]
        cls_score = torch.cat(cls_scores, dim=1)
        bbox_pred = torch.cat(bbox_preds, dim=1)
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # pdb.set_trace()

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels

    def simple_test_norsup(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            gt_labels: List[Tensor],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        softmax_flag = False
        gt_supclass_cls_ = self.conver_cls(gt_labels[0][0])
        subclass = self.get_subclass(gt_supclass_cls_)
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)

        rois = bbox2roi(proposals)
        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)

        support_key = []
        support_feats = []
        for key in support_feats_dict.keys():
            support_feats.append(support_feats_dict[key])
            support_key.append(key)

        support_feats = torch.cat(support_feats, 0)
        if self.use_gnn:
            support_feats, query_roi_feats = self.gnn._process_per_class(rois, query_roi_feats, support_feats)

        for i in range(support_feats.shape[0]):
            support_feats_dict[i] = support_feats[i, :].view(1, support_feats.shape[-1])

        supcls_scores = self.bbox_head.forward_supcls_(query_roi_feats.squeeze(-1).squeeze(-1))
        if softmax_flag:
            supcls_scores = F.softmax(supcls_scores, dim=-1)

        supcls_scores = F.softmax(supcls_scores, dim=-1) if supcls_scores is not None else None
        value, index_ = torch.max(supcls_scores, 1)

        gt_supclass_cls = []
        for c in index_:
            cl = c.item()
            if cl in gt_supclass_cls:
                continue
            gt_supclass_cls.append(cl)
        gt_supclass_cls = torch.tensor(gt_supclass_cls).cuda()
        subclass = self.get_subclass(gt_supclass_cls)

        cls_scores_dict, bbox_preds_dict = {}, {}
        num_classes = self.bbox_head.num_classes
        for class_id in support_feats_dict.keys():
            # pdb.set_trace()
            if class_id not in subclass:
                continue
            support_feat = support_feats_dict[class_id]
            bbox_results = self._bbox_forward(query_roi_feats, support_feat)
            if softmax_flag:
                bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)
            cls_scores_dict[class_id] = \
                bbox_results['cls_score'][:, class_id:class_id + 1]
            bbox_preds_dict[class_id] = \
                bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]

        for class_id in support_feats_dict.keys():

            support_feat = support_feats_dict[class_id]

            # sim_rate = torch.cosine_similarity(support_feats_dict[4], query_roi_feats, 1)
            bbox_results = self._bbox_forward(query_roi_feats, support_feat)

            if softmax_flag:
                bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)

            if cls_scores_dict.get(num_classes, None) is None:
                cls_scores_dict[num_classes] = \
                    bbox_results['cls_score'][:, -1:]
            else:
                cls_scores_dict[num_classes] += \
                    bbox_results['cls_score'][:, -1:]

        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())

        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
            for i in range(num_classes + 1)
        ]
        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
            for i in range(num_classes)
        ]
        cls_score = torch.cat(cls_scores, dim=1)
        bbox_pred = torch.cat(bbox_preds, dim=1)
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels

    def simple_test_bboxes(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            gt_labels: List[Tensor],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        softmax_flag = False
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)
        # pdb.set_trace()
        # weight_p = self.bbox_head.fc_cls.weight
        # torch.var(weight_p)
        # torch.mean(weight_p)
        rois = bbox2roi(proposals)
        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)

        support_key = []
        support_feats = []
        for key in support_feats_dict.keys():
            support_feats.append(support_feats_dict[key])
            support_key.append(key)

        support_feats = torch.cat(support_feats, 0)

        if self.use_gnn:
            support_feats, query_roi_feats = self.gnn._process_per_class(rois, query_roi_feats, support_feats)
        # pdb.set_trace()

        for i in range(support_feats.shape[0]):
            support_feats_dict[i] = support_feats[i, :].view(1, support_feats.shape[-1])

        cls_scores_dict, bbox_preds_dict = {}, {}
        num_classes = self.bbox_head.num_classes
        for class_id in support_feats_dict.keys():
            support_feat = support_feats_dict[class_id]
            bbox_results = self._bbox_forward(query_roi_feats, support_feat)
            if softmax_flag:
                bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)
            cls_scores_dict[class_id] = \
                bbox_results['cls_score'][:, class_id:class_id + 1]
            bbox_preds_dict[class_id] = \
                bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]

        for class_id in support_feats_dict.keys():

            support_feat = support_feats_dict[class_id]

            # sim_rate = torch.cosine_similarity(support_feats_dict[4], query_roi_feats, 1)
            bbox_results = self._bbox_forward(query_roi_feats, support_feat)

            if softmax_flag:
                bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)

            if cls_scores_dict.get(num_classes, None) is None:
                cls_scores_dict[num_classes] = \
                    bbox_results['cls_score'][:, -1:]
            else:
                cls_scores_dict[num_classes] += \
                    bbox_results['cls_score'][:, -1:]

        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())
        # cls_scores_dict[num_classes] = cls_scores_dict[num_classes] * 0.8

        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
            for i in range(num_classes + 1)
        ]
        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
            for i in range(num_classes)
        ]
        cls_score = torch.cat(cls_scores, dim=1)
        bbox_pred = torch.cat(bbox_preds, dim=1)

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels

    def simple_test_bboxes_text(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            gt_labels: List[Tensor],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        if self.with_text_sim_aug:
            softmax_flag = False
            img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
            scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)
            rois = bbox2roi(proposals)
            query_roi_feats = self.extract_query_roi_feat(query_feats, rois)

            support_key = []
            support_feats = []
            support_feats_dict_temp = copy.deepcopy(support_feats_dict)
            for key in support_feats_dict.keys():
                support_feats.append(support_feats_dict[key][0])
                support_key.append(key)

            support_feats = torch.cat(support_feats, 0)

            if self.use_gnn:
                support_feats, query_roi_feats = self.gnn._process_per_class(rois, query_roi_feats, support_feats)

            for i in range(support_feats.shape[0]):
                support_feats_dict[i] = support_feats[i, :].view(1, support_feats.shape[-1])

            cls_scores_dict, bbox_preds_dict = {}, {}
            num_classes = self.bbox_head.num_classes
            for class_id in support_feats_dict.keys():
                support_feat = support_feats_dict[class_id]
                bbox_results = self._bbox_forward(query_roi_feats, support_feat, support_feats_dict_temp[class_id][-1])
                if softmax_flag:
                    bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)
                cls_scores_dict[class_id] = \
                    bbox_results['cls_score'][:, class_id:class_id + 1]
                bbox_preds_dict[class_id] = \
                    bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]

            for class_id in support_feats_dict.keys():

                support_feat = support_feats_dict[class_id]

                # sim_rate = torch.cosine_similarity(support_feats_dict[4], query_roi_feats, 1)
                bbox_results = self._bbox_forward(query_roi_feats, support_feat)

                if softmax_flag:
                    bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)

                if cls_scores_dict.get(num_classes, None) is None:
                    cls_scores_dict[num_classes] = \
                        bbox_results['cls_score'][:, -1:]
                else:
                    cls_scores_dict[num_classes] += \
                        bbox_results['cls_score'][:, -1:]

            cls_scores_dict[num_classes] /= len(support_feats_dict.keys())

            cls_scores_dict[num_classes] = cls_scores_dict[num_classes]

            cls_scores = [
                cls_scores_dict[i] if i in cls_scores_dict.keys() else
                torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
                for i in range(num_classes + 1)
            ]
            bbox_preds = [
                bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
                torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
                for i in range(num_classes)
            ]
            cls_score = torch.cat(cls_scores, dim=1)
            bbox_pred = torch.cat(bbox_preds, dim=1)

            num_proposals_per_img = tuple(len(p) for p in proposals)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)

            # apply bbox post-processing to each image individually
            det_bboxes = []
            det_labels = []
            for i in range(len(proposals)):
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
        else:
            det_bboxes, det_labels = self.simple_test_bboxes(
                query_feats,
                support_feats_dict,
                query_img_metas,
                gt_labels,
                proposals,
                rcnn_test_cfg,
                rescale)
        return det_bboxes, det_labels

    def simple_dummpy(
            self,
            query_feats: List[Tensor],
            proposals: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        rois = bbox2roi(proposals)
        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)

        cls_scores_dict, bbox_preds_dict = {}, {}
        for class_id in support_feats_dict.keys():
            support_feat = support_feats_dict[class_id]
            bbox_results = self._bbox_forward(query_roi_feats, support_feat)
            if softmax_flag:
                bbox_results['cls_score'] = F.softmax(bbox_results['cls_score'], dim=-1)
            cls_scores_dict[class_id] = \
                bbox_results['cls_score'][:, class_id:class_id + 1]
            bbox_preds_dict[class_id] = \
                bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]

        return det_bboxes, det_labels
