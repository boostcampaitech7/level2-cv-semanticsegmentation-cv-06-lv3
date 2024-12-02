_base_ = [
    '../custom_models/custom_segformer.py',
    '../_base_/datasets/handbone_one_class_224x224.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]


crop_size = (224, 224)
data_preprocessor = dict(size=crop_size)
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
# model = dict(
#     data_preprocessor=data_preprocessor,
#     backbone=dict(
#         # init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
#         ),
#     # test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))
#     )

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # img_size = (2048,2048),
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
        # type='AdamW', lr=0.0006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=200),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=200,
        # end=160000,
        end=3000,
        by_epoch=False,
    )
]

# val_evaluator = dict(type='DiceCoefficient', num_classes=29)
val_evaluator = dict(type='IoUMetric')
test_evaluator = dict(
    type='LREOneMetric',
    output_dir='work_dirs/seg_b5_triquetrum_crop')

train_dataloader = dict(batch_size=32, num_workers=4)
val_dataloader = dict(batch_size=32, num_workers=4)
test_dataloader = dict(batch_size=32, num_workers=4)

# test_dataloader = val_dataloader
