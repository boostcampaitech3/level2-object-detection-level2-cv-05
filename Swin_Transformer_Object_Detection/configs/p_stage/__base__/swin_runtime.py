evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')
checkpoint_config = dict(max_keep_ckpts=2, interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ]
)
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
