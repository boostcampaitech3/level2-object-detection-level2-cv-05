_base_ = [
    "./__base__/cascade_rcnn_swin_fpn.py",
    "./__base__/coco_trash_dataset.py",
    './__base__/swin_scheduler.py',
    "./__base__/swin_runtime.py"
]

model_name = 'swin'
model_size = 'small'
model_loss = 'GIoU'
model_input = 224

model = dict(
    type='CascadeRCNN',
    pretrained="/opt/ml/detection/baseline/Swin-Transformer-Object-Detection/swin_small_patch4_window7_224.pth",
    backbone=dict(
        type='SwinTransformer',
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.3,
        ape=False,
        use_checkpoint=True),
    neck=dict(in_channels=[96, 96 * 2, 96 * 4, 96 * 8]),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
)

runner = dict(max_epochs=12)
classwise = True

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MlflowLoggerHook',
             exp_name=f'{model_name}_{model_size}_{model_loss}_{model_input}',
             tags=dict(
                 epochs=12,
                 optim='AdamW',
                 bbox_loss=model_loss,
                 rpn_loss='LabelSmoothing'
             ),
             )
    ]
)

work_dir = f'./work_dirs/{model_name}/{model_size}_{model_loss}_{model_input}'
seed = 2022
gpu_ids = 0
