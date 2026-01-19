# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
import copy

from mmdet.models.backbones import ResNet
from mmdet.models.builder import SHARED_HEADS
from mmdet.models.utils import ResLayer as _ResLayer
import pdb
from torch import Tensor


@SHARED_HEADS.register_module()
class DisMetaRCNNResLayer(BaseModule):

    def __init__(self,
                 depth,
                 stage=3,
                 stride=2,
                 dilation=1,
                 style='pytorch',
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 dcn=None,
                 pretrained=None,
                 init_cfg=None,
                 base_alpha=0.5,
                 loss_kd_weight=0.025,
                 base_cpt='/home/f523/disk1/sxp/mmfewshot/work_dirs/cls-meta-rcnn_r101_c4_8xb4_base-training_dior_split1/iter_60000.pth'):
        super(DisMetaRCNNResLayer, self).__init__(init_cfg)

        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.stage = stage
        self.fp16_enabled = False
        block, stage_blocks = ResNet.arch_settings[depth]
        stage_block = stage_blocks[stage]
        planes = 64 * 2 ** stage
        inplanes = 64 * 2 ** (stage - 1) * block.expansion

        res_layer1 = _ResLayer(
            block,
            inplanes,
            planes,
            stage_block,
            stride=stride,
            dilation=dilation,
            style=style,
            with_cp=with_cp,
            norm_cfg=self.norm_cfg,
            dcn=dcn)
        self.add_module(f'layer{stage + 1}', res_layer1)

        res_layer2 = _ResLayer(
            block,
            inplanes,
            planes,
            stage_block,
            stride=stride,
            dilation=dilation,
            style=style,
            with_cp=with_cp,
            norm_cfg=self.norm_cfg,
            dcn=dcn)
        self.add_module(f'layer{stage + 2}', res_layer2)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.max_pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()
        self.base_res_layer = getattr(self, f'layer{self.stage + 1}')
        self.novel_res_layer = getattr(self, f'layer{self.stage + 2}')

        self.base_alpha = base_alpha
        self.loss_kd = dict()
        self.loss_kd_weight = loss_kd_weight
        self.base_cpt = base_cpt

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        torch.manual_seed(0)
        processing = True
        # During training, processing the checkpoints
        # During testing, directly load the checkpoints
        for param_name in state_dict:
            if 'novel_res_layer' in param_name:
                processing = False
                break
        print('-------processing-----', processing)

        if processing:
            # load base training checkpoint
            base_weights = torch.load(self.base_cpt, map_location='cpu')
            if 'state_dict' in base_weights:
                base_weights = base_weights['state_dict']

            for n, p in base_weights.items():
                if 'shared_head' not in n:
                    state_dict[n] = copy.deepcopy(p)

            # initialize the base branch with the weight of base training
            for n, p in base_weights.items():
                if 'shared_head' in n:
                    new_n_base = n.replace('shared_head.layer4', 'shared_head.base_res_layer')
                    state_dict[new_n_base] = copy.deepcopy(p)
                    new_n_novel = n.replace('shared_head.layer4', 'shared_head.novel_res_layer')
                    state_dict[new_n_novel] = copy.deepcopy(p)
            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)


    def forward_single(self, x: Tensor) -> Tensor:
        """Forward function for query images.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).

        Returns:
            Tensor: Shape of (N, C).
        """
        # self.novel_res_layer[1].conv1.weight-base_weights['roi_head.shared_head.layer4.1.conv1.weight'].cuda()
        kd_loss_list = []
        alpha = self.base_alpha

        for name, param in self.base_res_layer.named_parameters():
            param.requires_grad = False

        base_x = self.base_res_layer(x)
        novel_x = self.novel_res_layer(x)

        out = alpha * base_x + (1 - alpha) * novel_x
        kd_loss_list.append(torch.frobenius_norm(base_x - out, dim=-1))

        kd_loss = torch.cat(kd_loss_list, dim=0)
        kd_loss = torch.mean(kd_loss)
        out = out.mean(3).mean(2)
        if self.training:
            kd_loss = kd_loss * self.loss_kd_weight
            self.loss_kd['loss_kd'] = kd_loss
        return out

    def forward_support_single(self, x: Tensor) -> Tensor:
        """Forward function for support images.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).

        Returns:
            Tensor: Shape of (N, C).
        """
        kd_loss_list = []
        x = self.max_pool(x)
        alpha = self.base_alpha
        base_x = self.base_res_layer(x)
        novel_x = self.novel_res_layer(x)
        x = alpha * base_x + (1 - alpha) * novel_x
        kd_loss_list.append(torch.frobenius_norm(base_x - x, dim=-1))
        out = self.sigmoid(x)

        kd_loss = torch.cat(kd_loss_list, dim=0)
        kd_loss = torch.mean(kd_loss)
        out = out.mean(3).mean(2)

        if self.training:
            kd_loss = kd_loss * self.loss_kd_weight
            self.loss_kd['loss_kd'] += kd_loss
        return out

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for query images.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).

        Returns:
            Tensor: Shape of (N, C).
        """
        # self.novel_res_layer[1].conv1.weight-base_weights['roi_head.shared_head.layer4.1.conv1.weight'].cuda()
        kd_loss_list = []
        alpha = self.base_alpha

        for name, param in self.base_res_layer.named_parameters():
            param.requires_grad = False

        assert len(self.base_res_layer) == len(self.novel_res_layer)
        for ind in range(len(self.base_res_layer)):
            base_x = self.base_res_layer[ind](x)
            novel_x = self.novel_res_layer[ind](x)
            x = alpha * base_x + (1 - alpha) * novel_x
            kd_loss_list.append(torch.frobenius_norm(base_x.mean(3).mean(2) - x.mean(3).mean(2), dim=-1))
        kd_loss = torch.cat(kd_loss_list, dim=0)
        kd_loss = torch.mean(kd_loss)
        out = x.mean(3).mean(2)
        if self.training:
            kd_loss = kd_loss * self.loss_kd_weight
            self.loss_kd['loss_kd'] = kd_loss
        return out

    def forward_support(self, x: Tensor) -> Tensor:
        """Forward function for support images.

        Args:
            x (Tensor): Features from backbone with shape (N, C, H, W).

        Returns:
            Tensor: Shape of (N, C).
        """
        kd_loss_list = []
        x = self.max_pool(x)
        alpha = self.base_alpha
        assert len(self.base_res_layer) == len(self.novel_res_layer)
        for ind in range(len(self.base_res_layer)):
            base_x = self.base_res_layer[ind](x)
            novel_x = self.novel_res_layer[ind](x)
            x = alpha * base_x + (1 - alpha) * novel_x
            kd_loss_list.append(torch.frobenius_norm(base_x.mean(3).mean(2) - x.mean(3).mean(2), dim=-1))

        out = self.sigmoid(x)

        kd_loss = torch.cat(kd_loss_list, dim=0)
        kd_loss = torch.mean(kd_loss)
        out = out.mean(3).mean(2)

        if self.training:
            kd_loss = kd_loss * self.loss_kd_weight
            self.loss_kd['loss_kd'] += kd_loss
        return out

    def train(self, mode=True):
        super(DisMetaRCNNResLayer, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

