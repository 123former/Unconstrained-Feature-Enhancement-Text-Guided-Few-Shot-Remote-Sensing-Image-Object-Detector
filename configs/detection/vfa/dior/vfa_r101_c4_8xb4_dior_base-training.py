_base_ = [
    './dior_split5_base.py',
    '../../_base_/schedules/schedule.py', '../vfa_r101_c4.py',
    '../../_base_/default_runtime.py'
]
lr_config = dict(warmup_iters=1000, step=[60000, 70000])
evaluation = dict(
    interval=20000,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=10000)
runner = dict(max_iters=80000)
optimizer = dict(lr=0.005)
# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=15, num_meta_classes=15)))
