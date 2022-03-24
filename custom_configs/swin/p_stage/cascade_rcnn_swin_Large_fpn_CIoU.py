_base_ = [
    "./__base__/cascade_rcnn_swin_fpn.py",
    "./__base__/coco_trash_dataset.py",
    './__base__/swin_scheduler.py',
    "./__base__/swin_runtime.py",
]

model_name = 'swin'
model_size = 'Large'
model_loss = 'CIoU'
model_input = 224

model = dict(
    type='CascadeRCNN',
    pretrained="/opt/ml/detection/baseline/Swin-Transformer-Object-Detection/swin_large_patch4_window7_224_22k.pth",
    backbone=dict(
        type='SwinTransformer',
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        drop_path_rate=0.3,
        ape=False,
        use_checkpoint=True),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    rpn_head=dict(
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
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
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
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
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
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
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='CIoULoss', loss_weight=10.0))
        ]
    )
)

runner = dict(max_epochs=12)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MlflowLoggerHook',
             exp_name=f'{model_name}_{model_size}_{model_loss}_{model_input}_base',
             tags=dict(
                 epochs=12,
                 optim='AdamW',
                 bbox_loss=model_loss,
                 rpn_loss='LabelSmoothing'
             ),
             )
    ]
)

work_dir = f'./work_dirs/{model_name}/{model_size}_{model_loss}_{model_input}_base'
seed = 2022
gpu_ids = 0
