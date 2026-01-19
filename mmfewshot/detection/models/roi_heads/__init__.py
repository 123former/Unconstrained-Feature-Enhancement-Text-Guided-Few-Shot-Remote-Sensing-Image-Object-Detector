# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_heads import (ContrastiveBBoxHead, CosineSimBBoxHead,
                         MultiRelationBBoxHead)
from .contrastive_roi_head import ContrastiveRoIHead
from .fsdetview_roi_head import FSDetViewRoIHead
from .meta_rcnn_roi_head import MetaRCNNRoIHead
from .multi_relation_roi_head import MultiRelationRoIHead
from .shared_heads import MetaRCNNResLayer
from .two_branch_roi_head import TwoBranchRoIHead
from .meta_rrcnn_roi_head import MetaRRCNNRoIHead
from .standard_rroi_head import StandardRRoIHead
from .meta_rcnn_roi_head_relation import MetaRCNNRelationRoIHead
from .multi_case import coco_reason_cls, COCO_SPLIT, voc_reason_cls, VOC_SPLIT
from .aug_meta_rcnn_roi_head import AugMetaRCNNRoIHead
from .fsce_roi_head import FscedRoIHead
from .gnn_meta_rcnn_roi_head import GNNMetaRCNNRoIHead
from .cls_meta_rcnn_roi_head import ClsMetaRCNNRoIHead
from .sentence import Text_Embedding
from .vfa_roi_head import VFARoIHead
from .cls_cat_roi_head import ClsRoIHead
from .vis_map import generstae_featmap
__all__ = [
    'CosineSimBBoxHead', 'ContrastiveBBoxHead', 'MultiRelationBBoxHead',
    'ContrastiveRoIHead', 'MultiRelationRoIHead', 'FSDetViewRoIHead',
    'MetaRCNNRoIHead', 'MetaRCNNResLayer', 'TwoBranchRoIHead', 'MetaRRCNNRoIHead',
    'StandardRRoIHead', 'MetaRCNNRelationRoIHead', 'coco_reason_cls', 'COCO_SPLIT',
    'AugMetaRCNNRoIHead', 'FscedRoIHead', 'GNNMetaRCNNRoIHead', 'ClsMetaRCNNRoIHead',
    'voc_reason_cls', 'VOC_SPLIT', 'Text_Embedding', 'VFARoIHead', 'ClsRoIHead', 'generstae_featmap'
]
