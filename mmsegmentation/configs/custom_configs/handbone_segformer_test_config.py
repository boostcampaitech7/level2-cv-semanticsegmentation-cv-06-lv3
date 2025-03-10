_base_ = [
    '../custom_models/custom_segformer.py',
    '../_base_/datasets/handbone_29_classes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_25k.py'
]
crop_size = (2048, 2048)
data_preprocessor = dict(size=crop_size)
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
        ),
    test_cfg=dict(mode='slide', crop_size=(2048, 2048), stride=(1024, 1024)))

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

train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2560, 1024), keep_ratio=True),
    # dict(type='Resize', scale=(2560, 1024), keep_ratio=True),
    dict(type='Resize', scale=(2048,2048), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]


dataset_type = 'XRayDataset'
data_root = '/data/ephemeral/home/level2-cv-semanticsegmentation-cv-06-lv3/data'

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='fold_test/images',
            #img_path='fold_test/images',
            seg_map_path=''),
        pipeline=test_pipeline))

test_evaluator = dict(
    type='LREMetric',
    output_dir='work_dirs/format_results'

)

val_evaluator = test_evaluator
