_base_ = [
    './_base_/schedules/schedule.py',
    './_base_/models/faster_rcnn_r50_caffe_fpn.py',
    './_base_/default_runtime.py'
]
# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(400, 400),
        keep_ratio=True,
        multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(400, 400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# classes splits are predefined in FewShotVOCDataset
data_root = 'data/NWPUv2/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='FewShotNWPUV2Dataset',
        save_dataset=False,
        ann_cfg=[
            dict(
                type='ann_file', ann_file='data/NWPUv2/NWPU2017/ImageSets/Main/trainval.txt')
        ],
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes='BASE_CLASSES_SPLIT1',
        use_difficult=True,
        instance_wise=False),
    val=dict(
        type='FewShotNWPUV2Dataset',
        ann_cfg=[
            dict(
                type='ann_file', ann_file='data/NWPUv2/NWPU2017/ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes='BASE_CLASSES_SPLIT1',
    ),
    test=dict(
        type='FewShotNWPUV2Dataset',
        ann_cfg=[
            dict(
                type='ann_file', ann_file='data/NWPUv2/NWPU2017/ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes='BASE_CLASSES_SPLIT1',
    ))
evaluation = dict(interval=2000, metric='mAP', class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# classes splits are predefined in FewShotVOCDataset
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24000, 32000])
runner = dict(type='IterBasedRunner', max_iters=36000)
# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(bbox_head=dict(num_classes=7)))
# using regular sampler can get a better base model
use_infinite_sampler = False
