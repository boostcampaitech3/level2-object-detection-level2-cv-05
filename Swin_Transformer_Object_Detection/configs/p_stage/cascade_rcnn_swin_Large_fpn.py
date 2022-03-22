_base_ = [
    "./__base__/cascade_rcnn_swin_fpn.py",
    "./__base__/coco_trash_dataset.py",
    './__base__/swin_scheduler.py'
    "./__base__/swin_runtime.py",
]

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
)

runner = dict(max_epochs=12)
work_dir = './work_dirs/swin/large_224'
seed = 2022
gpu_ids = 0
