_base_ = [
    "./__base__/cascade_rcnn_swin_fpn.py",
    "./__base__/coco_trash_dataset.py",
    './__base__/swin_scheduler.py',
    "./__base__/swin_runtime.py"
]

model = dict(
    type='CascadeRCNN',
    pretrained="/opt/ml/detection/baseline/Swin-Transformer-Object-Detection/swin_base_patch4_window7_224_22k.pth",
    backbone=dict(
        type='SwinTransformer',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.3,
        ape=False,
        use_checkpoint=True),
    neck=dict(in_channels=[128, 256, 512, 1024]),
)

runner = dict(max_epochs=12)
work_dir = './work_dirs/swin/base_224'
seed = 2022
gpu_ids = 0
