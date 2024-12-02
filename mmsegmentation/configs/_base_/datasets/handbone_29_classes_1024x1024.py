# dataset settings
dataset_type = 'XRayDataset'
data_root = '/data/ephemeral/home/level2-cv-semanticsegmentation-cv-06-lv3/data'
crop_size = (2048, 2048)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='LoadXRayAnnotations'),
    # dict(
    #     type='RandomResize',
    #     scale=(2560, 2048),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    dict(type='Resize', scale=(2048,2048), keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), #TODO too slow
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2560, 2048), keep_ratio=True),
    # dict(type='Resize', scale=(2560, 2048), keep_ratio=True),
    dict(type='Resize', scale=(2048,2048), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='LoadXRayAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            # Resize 옵션이 필요하다면 여기에 추가
            # [
            #     dict(type='Resize', scale_factor=r, keep_ratio=True)
            #     for r in img_ratios
            # ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [
                dict(type='PhotoMetricDistortion',
                     brightness_delta=0,  # 밝기 조정 없음 (원본 유지)
                     contrast_range=(1.0, 1.0)),  # 대비 조정 없음 (원본 유지)
                dict(type='PhotoMetricDistortion',
                     brightness_delta=16,  # 밝기 조정
                     contrast_range=(0.8, 1.2))  # 대비 조정
            ],
            # 원본 데이터
            # [
            #     dict(type='RandomRotate', prob=0., degree=5),  # 회전 없음
            #     dict(type='RandomRotate', prob=1.0, degree=5)  # ±5° 회전
            # ],
  
            [
                dict(type='PackSegInputs')  # Annotation 로드 제거
            ]
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='fold_0/images', seg_map_path='fold_0/annos'),
        pipeline=train_pipeline))


val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='fold_1/images',
            seg_map_path='fold_1/annos'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

#테스트용 주석
# val_evaluator = dict(type='DiceCoefficient', num_classes=29)
# test_evaluator = val_evaluator
