#1
python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/ablation/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.py

python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/ablation/meta-rcnn_r101_c4_8xb4_voc-split1_20shot-fine-tuning.py

python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/ablation/meta-rcnn_r101_c4_8xb4_voc-split1_10shot-fine-tuning.py
# 2
python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/ablation/meta-rcnn_r101_c4_8xb4_voc-split1_20shot-fine-tuning_nof.py

python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/ablation/meta-rcnn_r101_c4_8xb4_voc-split1_10shot-fine-tuning_nof.py
# 3
python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/ablation/meta-rcnn_r101_c4_8xb4_voc-split1_20shot-fine-tuning_nof_dis.py

python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/ablation/meta-rcnn_r101_c4_8xb4_voc-split1_10shot-fine-tuning_nof_dis.py

# 4
python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/ablation/cls-meta-rcnn_r101_c4_8xb4_dior-split1_10shot-fine-tuning_ngnn_plus.py

# 7
python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/ablation/cls-meta-rcnn_r101_c4_8xb4_10shot-fine-tuning_dior_split1.py

# 5
python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/ablation/meta-rcnn_r101_c4_8xb4_voc-split1_base-training_ngnn_plus_augtext_mask.py

python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/ablation/meta-rcnn_r101_c4_8xb4_voc-split1_20shot-fine-tuning_ngnn_plus_augtext_mask.py

python ./tools/detection/train.py /home/f523/disk1/sxp/mmfewshot/configs/detection/ablation/meta-rcnn_r101_c4_8xb4_voc-split1_10shot-fine-tuning_ngnn_plus_augtext_mask.py