epoch_num = 12
model = dict(
    type='CascadeRCNN',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='DetectoRS_ResNeXt',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='CIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='CIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='CIoULoss', loss_weight=10.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=100)))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')
checkpoint_config = dict(max_keep_ckpts=2, interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MlflowLoggerHook',
            exp_name='cascade_rcnn_ResNeXt_RFP',
            tags=dict(
                optim='AdamW',
                bbox_loss='CIoU',
                rpn_loss='LabelSmoothing',
                fold=4))
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
size_min = 512
size_max = 1536
multi_scale = [(512, 512), (544, 544), (576, 576), (608, 608), (640, 640),
               (672, 672), (704, 704), (736, 736), (768, 768), (800, 800),
               (832, 832), (864, 864), (896, 896), (928, 928), (960, 960),
               (992, 992), (1024, 1024), (1056, 1056), (1088, 1088),
               (1120, 1120), (1152, 1152), (1184, 1184), (1216, 1216),
               (1248, 1248), (1280, 1280), (1312, 1312), (1344, 1344),
               (1376, 1376), (1408, 1408), (1440, 1440), (1472, 1472),
               (1504, 1504), (1536, 1536)]
multi_scale_test = [(512, 512), (768, 768), (1024, 1024), (1280, 1280),
                    (1536, 1536)]
multi_scale_light = [(512, 512), (768, 768), (1024, 1024)]
img_norm_cfg = dict(
    mean=[122.6902, 116.4859, 109.2194],
    std=[60.9837, 59.9108, 61.882],
    to_rgb=True)
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic',
           'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
albu_transform = [
    dict(type='VerticalFlip', p=0.1),
    dict(type='HorizontalFlip', p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussNoise', p=1.0),
            dict(type='GaussianBlur', p=1.0),
            dict(type='Blur', p=1.0)
        ],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ShiftScaleRotate', p=1.0),
            dict(type='RandomRotate90', p=1.0)
        ],
        p=0.1),
    dict(
        type='ColorJitter',
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5,
        p=0.1)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(512, 512), (544, 544), (576, 576), (608, 608), (640, 640),
                   (672, 672), (704, 704), (736, 736), (768, 768), (800, 800),
                   (832, 832), (864, 864), (896, 896), (928, 928), (960, 960),
                   (992, 992), (1024, 1024), (1056, 1056), (1088, 1088),
                   (1120, 1120), (1152, 1152), (1184, 1184), (1216, 1216),
                   (1248, 1248), (1280, 1280), (1312, 1312), (1344, 1344),
                   (1376, 1376), (1408, 1408), (1440, 1440), (1472, 1472),
                   (1504, 1504), (1536, 1536)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(
        type='Albu',
        transforms=[
            dict(type='VerticalFlip', p=0.1),
            dict(type='HorizontalFlip', p=0.3),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='GaussNoise', p=1.0),
                    dict(type='GaussianBlur', p=1.0),
                    dict(type='Blur', p=1.0)
                ],
                p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='ShiftScaleRotate', p=1.0),
                    dict(type='RandomRotate90', p=1.0)
                ],
                p=0.1),
            dict(
                type='ColorJitter',
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5,
                p=0.1)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[122.6902, 116.4859, 109.2194],
        std=[60.9837, 59.9108, 61.882],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512), (768, 768), (1024, 1024), (1280, 1280),
                   (1536, 1536)],
        flip=False,
        transforms=[
            dict(
                type='Resize',
                img_scale=[(512, 512), (768, 768), (1024, 1024), (1280, 1280),
                           (1536, 1536)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[122.6902, 116.4859, 109.2194],
                std=[60.9837, 59.9108, 61.882],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
        ann_file='/opt/ml/detection/baseline/StratifiedKFold/cv_train_3.json',
        img_prefix='/opt/ml/detection/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(512, 512), (544, 544), (576, 576), (608, 608),
                           (640, 640), (672, 672), (704, 704), (736, 736),
                           (768, 768), (800, 800), (832, 832), (864, 864),
                           (896, 896), (928, 928), (960, 960), (992, 992),
                           (1024, 1024), (1056, 1056), (1088, 1088),
                           (1120, 1120), (1152, 1152), (1184, 1184),
                           (1216, 1216), (1248, 1248), (1280, 1280),
                           (1312, 1312), (1344, 1344), (1376, 1376),
                           (1408, 1408), (1440, 1440), (1472, 1472),
                           (1504, 1504), (1536, 1536)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(
                type='Albu',
                transforms=[
                    dict(type='VerticalFlip', p=0.1),
                    dict(type='HorizontalFlip', p=0.3),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='GaussNoise', p=1.0),
                            dict(type='GaussianBlur', p=1.0),
                            dict(type='Blur', p=1.0)
                        ],
                        p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='ShiftScaleRotate', p=1.0),
                            dict(type='RandomRotate90', p=1.0)
                        ],
                        p=0.1),
                    dict(
                        type='ColorJitter',
                        brightness=0.5,
                        contrast=0.5,
                        saturation=0.5,
                        hue=0.5,
                        p=0.1)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_bboxes='bboxes'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[122.6902, 116.4859, 109.2194],
                std=[60.9837, 59.9108, 61.882],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
        ann_file='/opt/ml/detection/baseline/StratifiedKFold/cv_val_3.json',
        img_prefix='/opt/ml/detection/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(512, 512), (768, 768), (1024, 1024)],
                flip=False,
                transforms=[
                    dict(
                        type='Resize',
                        img_scale=[(512, 512), (768, 768), (1024, 1024)],
                        multiscale_mode='value',
                        keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[122.6902, 116.4859, 109.2194],
                        std=[60.9837, 59.9108, 61.882],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
        ann_file='/opt/ml/detection/dataset/test.json',
        img_prefix='/opt/ml/detection/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(512, 512), (768, 768), (1024, 1024), (1280, 1280),
                           (1536, 1536)],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[122.6902, 116.4859, 109.2194],
                        std=[60.9837, 59.9108, 61.882],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)
optimizer = dict(
    type='AdamW',
    lr=5e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
    type='DistOptimizerHook',
    update_interval=1,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.1,
    min_lr_ratio=5e-06)
work_dir = './work_dirs/CNN/cascade_rcnn_ResNeXt_RFP'
seed = 2022
gpu_ids = range(0, 1)
