# configs/custom_configs/handbone_beit_config.py

# 기본 설정 파일들 로드
_base_ = [
    '../custom_models/custom_upernet_beit.py',  # 커스텀 모델 설정 파일
    '../_base_/datasets/handbone_29_classes_1024x1024.py',  # 커스텀 데이터셋 설정 파일
    '../_base_/default_runtime.py',  # 기본 런타임 설정
    # '../_base_/schedules/schedule_320k.py'  # 학습 스케줄 설정
    '../_base_/schedules/schedule_25k.py' 
]

# 데이터 전처리 설정
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)

# 모델 설정
model = dict(
    data_preprocessor=data_preprocessor,
    # pretrained='pretrain/beit_large_patch16_224_pt22k_ft22k.pth',  # 사전 학습된 가중치 경로
    backbone=dict(
        type='BEiT',
        img_size = (640,640),
    #     embed_dims=1024,
    #     num_layers=24,
    #     num_heads=16,
    #     mlp_ratio=4,
    #     qv_bias=True,
    #     init_values=1e-6,
    #     drop_path_rate=0.2,
    #     out_indices=[7, 11, 15, 23]
    ),
    # neck=dict(
    #     embed_dim=1024, 
    #     rescales=[4, 2, 1, 0.5]
    # ),
    # decode_head=dict(in_channels=[1024, 1024, 1024, 1024], num_classes=29, channels=1024),# 클래스 수 변경 
    # auxiliary_head=dict(in_channels=1024, num_classes=29),    # 클래스 수 변경 
    # test_cfg=dict(
    #     mode='slide', 
    #     crop_size=(640, 640), 
    #     stride=(426, 426)
    # )
)

# 최적화 설정
# optim_wrapper = dict(
#     _delete_=True,
#     type='AmpOptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.05
#     ),
#     constructor='LayerDecayOptimizerConstructor',
#     paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.95),
#     accumulative_counts=2
# )

# 학습률 스케줄러 설정
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=3000
#     ),
#     dict(
#         type='PolyLR',
#         power=1.0,
#         begin=3000,
#         end=160000,
#     eta_min=0.0,
#         by_epoch=False,
#     )
# ]

# 데이터 로더 설정
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    # dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    dict(type='Resize', scale=(640,640), keep_ratio=True),
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
            # img_path='fold_1/images',
            seg_map_path=''),
        pipeline=test_pipeline))

test_evaluator = dict(
    type='LREMetric',
    output_dir='work_dirs/format_results'

)

# test_dataloader = val_dataloader

# test_evaluator = dict(type='DiceCoefficient', num_classes=29)
# test_evaluator = dict(type='LREMetric',output_dir='./')

val_evaluator = test_evaluator

# 평가 지표 설정
# evaluation = dict(
#     interval=5,  # 검증 주기
#     metric=['DiceCoefficient'],  # 커스텀 Dice 지표 사용
#     metric_options=dict(
#         DiceCoefficient=dict(
#             num_classes=29,
#             ignore_index=255
#         )
#     )
# )
