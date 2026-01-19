#!/usr/bin/env bash

##vfa
#voc
#base training
#CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split1/vfa_r101_c4_8xb4_voc-split1_base-training.py
##fine tuning 10shot
#CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split1/vfa_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py
#CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split1/vfa_r101_c4_8xb4_voc-split1_2shot-fine-tuning.py
#CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split1/vfa_r101_c4_8xb4_voc-split1_3shot-fine-tuning.py
#CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split1/vfa_r101_c4_8xb4_voc-split1_5shot-fine-tuning.py
#CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split1/vfa_r101_c4_8xb4_voc-split1_10shot-fine-tuning.py

#base training
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split2/vfa_r101_c4_8xb4_voc-split2_base-training.py
#fine tuning 10shot
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split2/vfa_r101_c4_8xb4_voc-split2_1shot-fine-tuning.py
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split2/vfa_r101_c4_8xb4_voc-split2_2shot-fine-tuning.py
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split2/vfa_r101_c4_8xb4_voc-split2_3shot-fine-tuning.py
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split2/vfa_r101_c4_8xb4_voc-split2_5shot-fine-tuning.py
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split2/vfa_r101_c4_8xb4_voc-split2_10shot-fine-tuning.py

#base training
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split3/vfa_r101_c4_8xb4_voc-split3_base-training.py
#fine tuning 10shot
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split3/vfa_r101_c4_8xb4_voc-split3_1shot-fine-tuning.py
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split3/vfa_r101_c4_8xb4_voc-split3_2shot-fine-tuning.py
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split3/vfa_r101_c4_8xb4_voc-split3_3shot-fine-tuning.py
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split3/vfa_r101_c4_8xb4_voc-split3_5shot-fine-tuning.py
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/voc/vfa_split3/vfa_r101_c4_8xb4_voc-split3_10shot-fine-tuning.py
##coco
##base training
#CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/coco/vfa_r101_c4_8xb4_coco_base-training.py
##fine tuning 10shot
#CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/vfa/coco/vfa_r101_c4_8xb4_coco_10shot-fine-tuning.py
