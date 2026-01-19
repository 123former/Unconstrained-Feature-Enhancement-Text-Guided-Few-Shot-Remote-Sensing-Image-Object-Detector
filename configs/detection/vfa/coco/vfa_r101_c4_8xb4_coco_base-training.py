_base_ = [
    '../../_base_/datasets/nway_kshot/base_coco_ms.py',
    '../../_base_/schedules/schedule.py', '../vfa_r101_c4.py',
    '../../_base_/default_runtime.py'
]
lr_config = dict(warmup_iters=1000, step=[680000, 800000])
evaluation = dict(interval=80000)
checkpoint_config = dict(interval=10000)
runner = dict(max_iters=880000)
optimizer = dict(lr=0.000625)
# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=60, num_meta_classes=60)))