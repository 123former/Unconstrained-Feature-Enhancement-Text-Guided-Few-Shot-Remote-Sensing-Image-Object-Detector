#meta rcnn + cat+gnn
#base training
#CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/cls-vfa/dior/cls_vfa_r101_c4_8xb4_coco_base-training.py
#fine tuning 20shot
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/cls-vfa/dior/cls_vfa_r101_c4_8xb4_coco_20shot-fine-tuning.py

CUDA_VISIBLE_DEVICES=1 python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/cls-vfa/dior/cls_vfa_r101_c4_8xb4_coco_20shot-fine-tuning_nfreeze.py