"""
Created on Wednesday, April 27, 2022

@author: Guangxing Han
"""
import pdb

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from mmdet.core import BboxOverlaps2D
from .cal_rcnn_res_layer import SelAttention
from .multi_relation_head import MultiRelationHead

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features=2048, out_features=2048, dropout=0.5, alpha=0.1, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):

        batch, channel = input.size(0), input.size(1)
        # input=input.mean(3).mean(2)
        h = torch.mm(input, self.W)  # shape [N, out_features]
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                          2 * self.out_features)  # shape[N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N,N,1] -> [N,N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GraphConvAttentionLayer(nn.Module):

    def __init__(self, in_features=2048, out_features=2048, dropout=0.5, alpha=0.1, concat=True):
        super(GraphConvAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.head = MultiRelationHead()

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):

        batch, channel = input.size(0), input.size(1)
        h = torch.mm(input, self.W)  # shape [N, out_features]
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                          2 * self.out_features)  # shape[N, N, 2*out_features]

        # e = self.leakyrelu(self.head(a_input[:, :, :2048], a_input[:, :, 2048:]))  # [N,N,1] -> [N,N]
        # e_ = self.leakyrelu(e)
        # e_ = self.leakyrelu(e + torch.matmul(a_input, self.a).squeeze(2))

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        adj_ = torch.eye(adj.shape[0]).cuda()
        attention = torch.where(adj_ > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime



    def loss(self, cls_scores, rois, labels, label_weights, support_gt_labels, reduction_override=None):
        loss_value = self.head.loss(cls_scores, rois, labels, label_weights, support_gt_labels, reduction_override)
        return loss_value


class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer as https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_conv = nn.Conv2d(in_features, out_features, 1, padding=0, bias=False)
        self.norm = nn.LayerNorm([out_features, 7, 7])

    def reset_parameters(self):
        nn.init.normal_(self.graph_conv.weight, std=0.01)
        nn.init.constant_(self.graph_conv.bias, 0)

    def forward(self, input, adj):  # input: B*2048*7*7,      adj: B*B

        batch, channel = input.size(0), input.size(1)
        input_norm = self.norm(input)
        tmp = self.graph_conv(input_norm)
        output = torch.mm(adj, tmp.view(batch, -1)).view(batch, self.out_features, 7, 7) + input
        return output.mean(3).mean(2)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GNN(torch.nn.Module):
    def __init__(self, gnn_config='GAT'):
        super(GNN, self).__init__()

        self.feat_dim = 2048
        self.child_num = 4
        self.iou_threshold = 0.7
        self.child_iou_num = 10  # 8
        if gnn_config not in ['GCN', 'GAT', 'GCAT']:
            print('gnn_config is error!')
            pdb.set_trace()
        self.gnn_config = gnn_config

        # self.c2b = SelAttention(2048*2, 256, 2048*2, out_dim=1, Sigmoid_flag=True)

        if gnn_config == 'GCN':
            self.gcn_layer = GraphConvolutionLayer(self.feat_dim, self.feat_dim)
        elif gnn_config == 'GCAT':
            self.gcn_layer = GraphConvAttentionLayer(self.feat_dim, self.feat_dim)
        else:
            self.gcn_layer = GraphAttentionLayer(self.feat_dim, self.feat_dim)

    def forward(self, input_features, adj_mat=None):
        batch, channel = input_features.size(0), input_features.size(1)
        input_features_reshape = input_features.view(batch, -1).contiguous()

        # cosine similarity
        dot_product_mat = torch.mm(input_features_reshape, torch.transpose(input_features_reshape, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(input_features_reshape * input_features_reshape, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        cos_sim_mat = dot_product_mat / len_mat

        if adj_mat is not None:
            adj_mat = adj_mat.to(cos_sim_mat.device)
            new_adj_mat = adj_mat * cos_sim_mat
        else:
            new_adj_mat = cos_sim_mat.cuda()

        # new_adj_mat = adj_mat.cuda()
        gcn_ft = self.gcn_layer(input_features, new_adj_mat)
        return gcn_ft

    def cal_IoU_distance_matrix(self, proposal_boxes):
        pairwise_iou = BboxOverlaps2D()
        IoU_matrix = pairwise_iou(proposal_boxes, proposal_boxes)  # .detach().cpu().numpy()
        return IoU_matrix

    def _sample_child_nodes(self, center_idx, IoU_matrix):
        # obtain iou array for all the proposals
        act_iou = IoU_matrix[center_idx, :]
        rm_act_iou = act_iou
        rm_act_iou[center_idx] = 0
        pos_iou_idx = torch.where(rm_act_iou > self.iou_threshold)

        if len(pos_iou_idx) != 0:
            pos_iou_arr = rm_act_iou[pos_iou_idx]
            sorted_pos_iou_idx = torch.argsort(-pos_iou_arr)
            selected_pos_iou_idx = sorted_pos_iou_idx.repeat(self.child_iou_num)
            ref_iou_idx = selected_pos_iou_idx[:self.child_iou_num]
            abs_iou_idx = pos_iou_idx[0][ref_iou_idx]
        else:
            abs_iou_idx = torch.tensor([center_idx]).repeat(self.child_iou_num)

        idx = torch.randperm(abs_iou_idx.nelement())
        abs_iou_idx = abs_iou_idx.view(-1)[idx].view(abs_iou_idx.size())
        abs_child_idx = abs_iou_idx[:self.child_num]
        return abs_child_idx

    def _process_per_class(self, proposals, query_features, support_feat, image_size=(0, 0)):
        # compute pairwise IoU and distance among all proposal pairs
        # proposal_boxes = torch.cat(proposals, 0)
        proposal_boxes = proposals[:, 1:]

        IoU_matrix = self.cal_IoU_distance_matrix(proposal_boxes)

        # calculate the neighborhood for each child
        plus_node_num = support_feat.shape[0]
        adj_mat = torch.zeros((len(proposal_boxes) + plus_node_num, len(proposal_boxes) + plus_node_num),
                              dtype=torch.float)

        for idx_ in range(len(proposal_boxes)):
            idxs = self._sample_child_nodes(idx_, IoU_matrix)
            for idx_tmp in idxs:
                adj_mat[idx_][idx_tmp] += 1
            adj_mat[idx_, :] /= float(self.child_num)

            adj_mat[idx_, len(proposal_boxes):(len(proposal_boxes) + plus_node_num)] = 1
            adj_mat[idx_, idx_] = 1

        for i in range(len(proposal_boxes), len(proposal_boxes) + plus_node_num):
            adj_mat[i, :len(proposal_boxes)] = float(1) / len(proposals)
            adj_mat[i, len(proposal_boxes):(len(proposal_boxes) + plus_node_num)] = 0
            adj_mat[i, i] = 1

        # pdb.set_trace()
        # add the prototype node, and also the corresponding edges
        # adj_mat[-1, :] = 0
        # adj_mat[-1, -1] = 1

        # build graph and use GCN

        box_features_initial = torch.cat([query_features, support_feat], dim=0)
        box_features_gcn = self.forward(box_features_initial, adj_mat)
        query_features_ = torch.cat([query_features, box_features_gcn[:len(proposal_boxes), :]], 1)
        support_feat_ = torch.cat([support_feat, box_features_gcn[len(proposal_boxes):, :]], 1)
        return support_feat_, query_features_
        # return box_features_gcn
