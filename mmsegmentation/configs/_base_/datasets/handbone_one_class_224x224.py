# dataset settings
dataset_type = 'XRayCropDataset'
data_root = '/data/ephemeral/home/level2-cv-semanticsegmentation-cv-06-lv3/data'
# crop_size = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='LoadXRayAnnotations'),
    # dict(
    #     type='RandomResize',
    #     scale=(2560, 640),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    # dict(type='Resize', scale=(640,640), keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), #TODO too slow
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    # dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    # dict(type='Resize', scale=(640,640), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='LoadXRayAnnotations'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    # dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    # dict(type='Resize', scale=(640,640), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='LoadXRayAnnotations'),
    dict(type='PackSegInputs')
]


# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
img_ratios = [0.5, 1.0, 1.5]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
        #     [
        #         dict(type='Resize', scale_factor=r, keep_ratio=True)
        #         for r in img_ratios
        #     ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], 
            # [
            #     dict(type='LoadAnnotations') 
            #     # dict(type='LoadXRayAnnotations')
            #     ], 
                [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            # img_path='crop_all_Trapezoid/images', 
            # seg_map_path='crop_all_Trapezoid/annos'),
            img_path='crop_all_Triquetrum/images', 
            seg_map_path='crop_all_Triquetrum/annos'),

        pipeline=train_pipeline))


val_dataloader = dict(
    batch_size= 32,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='crop_val_Triquetrum/images',
            seg_map_path='crop_val_Triquetrum/annos'),
        pipeline=val_pipeline))


test_dataloader = dict(
    batch_size= 32,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='crop_test_Triquetrum/images',
            seg_map_path=''),
        pipeline=test_pipeline))

# val_evaluator = dict(type='DiceCoefficient', num_classes=29)
# test_evaluator = val_evaluator
