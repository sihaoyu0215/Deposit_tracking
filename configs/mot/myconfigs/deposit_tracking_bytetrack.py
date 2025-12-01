img_scale = (1080, 1920)
model = dict(
    detector=dict(
        type='YOLOX',
        input_size=(1080, 1920),
        random_size_range=(18, 32),
        random_size_interval=10,
        backbone=dict(
            type='CSPDarknet', deepen_factor=1.33, widen_factor=1.25),
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[320, 640, 1280],
            out_channels=320,
            num_csp_blocks=4),
        bbox_head=dict(
            type='YOLOXHead',
            num_classes=1,
            in_channels=320,
            feat_channels=320),
        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'work_dir/deposit_tracking_bytetrack/epoch_50.pth'
        )),
    type='ByteTrack',
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))
dataset_type = 'MOTChallengeDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(
        type='Mosaic',
        img_scale=(1080, 1920),
        pad_val=114.0,
        bbox_clip_border=False),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-540, -960),
        bbox_clip_border=False),
    dict(
        type='MixUp',
        img_scale=(1080, 1920),
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=False),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=(1080, 1920),
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1080, 1920),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data_root = 'data/task_pipe_new/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file=[
                'data/task_pipe_new/annotations/half-train_cocoformat.json'
            ],
            img_prefix=['data/task_pipe_new/train'],
            classes=('Deposit', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=[
            dict(
                type='Mosaic',
                img_scale=(1080, 1920),
                pad_val=114.0,
                bbox_clip_border=False),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-540, -960),
                bbox_clip_border=False),
            dict(
                type='MixUp',
                img_scale=(1080, 1920),
                ratio_range=(0.8, 1.6),
                pad_val=114.0,
                bbox_clip_border=False),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Resize',
                img_scale=(1080, 1920),
                keep_ratio=True,
                bbox_clip_border=False),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='MOTChallengeDataset',
        ann_file='data/task_pipe_new/annotations/half-val_cocoformat.json',
        img_prefix='data/task_pipe_new/train',
        ref_img_sampler=None,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1080, 1920),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(
                        type='Pad',
                        size_divisor=32,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='VideoCollect', keys=['img'])
                ])
        ],
        interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)),
    test=dict(
        type='MOTChallengeDataset',
        ann_file='data/task_pipe_new/annotations/test_cocoformat.json',
        img_prefix='data/task_pipe_new/test',
        ref_img_sampler=None,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1080, 1920),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(
                        type='Pad',
                        size_divisor=32,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='VideoCollect', keys=['img'])
                ])
        ],
        interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)),
    persistent_workers=True)
optimizer = dict(
    type='SGD',
    lr=0.00025,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
samples_per_gpu = 2
total_epochs = 40
num_last_epochs = 10
interval = 5
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,
    num_last_epochs=10,
    min_lr_ratio=0.05)
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=10, priority=48),
    dict(type='SyncNormHook', num_last_epochs=10, interval=5, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
fp16 = dict(loss_scale=dict(init_scale=512.0))
work_dir = 'work_dir/deposit_tracking_bytetrack_revise'
seed = 0
gpu_ids = range(0, 1)
