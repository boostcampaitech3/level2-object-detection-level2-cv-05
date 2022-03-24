_base_ = [
    "./__base__/cascade_rcnn_swin_fpn.py",
    "./__base__/coco_trash_dataset.py",
    './__base__/swin_scheduler.py',
    "./__base__/swin_runtime.py"
]

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
)

runner = dict(max_epochs=12)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MlflowLoggerHook',
             exp_name='swin_small_224',
             tags=dict(
                 epochs=12,
                 optim='AdamW',
                 bbox_loss='GIoU',
                 rpn_loss='LabelSmoothing'
             ),
             )
    ]
)

work_dir = './work_dirs/swin/small_224'
seed = 2022
gpu_ids = 0
