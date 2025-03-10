_base_ = './handbone_deeplabv3plus_config2.py'
norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)

model = dict(
    # pretrained='open-mmlab://resnet101_v1c', 
    backbone=dict(norm_cfg=norm_cfg,
                  depth=101),
    decode_head=dict(norm_cfg=norm_cfg),  # Decode Head에 적용
    auxiliary_head=dict(norm_cfg=norm_cfg)  # Auxiliary Head에 적용
    )

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

dataset_type = 'XRayDataset'
data_root = '/data/ephemeral/home/level2-cv-semanticsegmentation-cv-06-lv3/data'

val_dataloader = dict(batch_size=1, num_workers=4)
val_evaluator = dict(type='DiceCoefficient', num_classes=29)
# test_evaluator = val_evaluator

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    # dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    dict(type='Resize', scale=(2048,2048), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='fold_test/images',
            # img_path='fold_1/images',
            seg_map_path=''),
        pipeline=test_pipeline))

test_evaluator = dict(
    type='LREMetric',
    output_dir='work_dirs/format_results'

)

# train_dataloader = dict(batch_size=1, num_workers=4)
# test_dataloader = val_dataloader

# fp16 = dict(loss_scale='dynamic')

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='WandbVisBackend',
             init_kwargs=dict(
                 project='mmsegmentation',
                 name='deeplabv3plus_25k',
                #  entity='your_wandb_username'
             ),
             save_dir='wandb_logs')
    ],
    name='visualizer'
)
