data_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
anno_root = '/opt/ml/detection/baseline/StratifiedKFold/'

image_size = 1024
size_min, size_max = map(int, (image_size * 0.5, image_size * 1.5))

multi_scale = [(x, x) for x in range(size_min, size_max+1, 64)]
multi_scale_light = [(512, 512),(768, 768),(1024, 1024)]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
           'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=multi_scale,
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=multi_scale_light,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=multi_scale_light, multiscale_mode='value', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=data_type,
        classes=classes,
        ann_file=anno_root + 'cv_train_1.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=data_type,
        classes=classes,
        ann_file=anno_root + 'cv_val_1.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=data_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline)
)
