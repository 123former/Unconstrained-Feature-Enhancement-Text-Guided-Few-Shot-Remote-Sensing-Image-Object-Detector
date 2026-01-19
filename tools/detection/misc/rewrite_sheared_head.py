import torch
import pdb
import copy

base_cpt = '/home/f523/disk1/sxp/mmfewshot/work_dirs/cls-meta-rcnn_r101_c4_8xb4_voc-split1_base-training_ngnn_plus_augtext_mask/iter_60000.pth'


def rewrite_weught(processing=True):
    state_dict = dict()
    if processing:
        # load base training checkpoint
        base_weights = torch.load(base_cpt, map_location='cpu')
        if 'state_dict' in base_weights:
            base_weights = base_weights['state_dict']

        for n, p in base_weights.items():
            if 'shared_head' not in n:
                state_dict[n] = copy.deepcopy(p)

        # initialize the base branch with the weight of base training
        for n, p in base_weights.items():
            if 'shared_head' in n:
                new_n_base = n.replace('shared_head', 'shared_head.base_res_layer')
                state_dict[new_n_base] = copy.deepcopy(p)
                new_n_novel = n.replace('shared_head', 'shared_head.novel_res_layer')
                state_dict[new_n_novel] = copy.deepcopy(p)

        base_weights['state_dict'] = state_dict
        out_path = '/home/f523/disk1/sxp/mmfewshot/work_dirs/cls-meta-rcnn_r101_c4_8xb4_voc-split1_base-training_ngnn_plus_augtext_mask/rewrite_weight.pth'
        torch.save(base_weights, out_path)


if __name__ == '__main__':
    rewrite_weught()
