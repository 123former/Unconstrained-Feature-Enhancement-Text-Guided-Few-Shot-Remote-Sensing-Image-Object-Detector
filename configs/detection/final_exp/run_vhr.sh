# dior vhr10
#python ./tools/detection/train.py configs/detection/final_exp/vhr10/split1/cls-meta-rcnn_r101_c4_8xb4_base-training_vhr10_split1.py
#
#python ./tools/detection/train.py configs/detection/final_exp/vhr10/split2/cls-meta-rcnn_r101_c4_8xb4_base-training_vhr10_split2.py
#
## vhr10 split1
#python ./tools/detection/train.py configs/detection/final_exp/vhr10/split1/cls-meta-rcnn_r101_c4_8xb4_20shot-fine-tuning_vhr10_split1.py
#
#python ./tools/detection/train.py configs/detection/final_exp/vhr10/split1/cls-meta-rcnn_r101_c4_8xb4_10shot-fine-tuning_vhr10_split1.py
#
#python ./tools/detection/train.py configs/detection/final_exp/vhr10/split1/cls-meta-rcnn_r101_c4_8xb4_5shot-fine-tuning_vhr10_split1.py
#
#python ./tools/detection/train.py configs/detection/final_exp/vhr10/split1/cls-meta-rcnn_r101_c4_8xb4_3shot-fine-tuning_vhr10_split1.py

# vhr10 split2
#CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py configs/detection/final_exp/vhr10/split2/cls-meta-rcnn_r101_c4_8xb4_20shot-fine-tuning_vhr10_split2.py

CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py configs/detection/final_exp/vhr10/split2/cls-meta-rcnn_r101_c4_8xb4_10shot-fine-tuning_vhr10_split2.py

CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py configs/detection/final_exp/vhr10/split2/cls-meta-rcnn_r101_c4_8xb4_5shot-fine-tuning_vhr10_split2.py

CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py configs/detection/final_exp/vhr10/split2/cls-meta-rcnn_r101_c4_8xb4_3shot-fine-tuning_vhr10_split2.py
