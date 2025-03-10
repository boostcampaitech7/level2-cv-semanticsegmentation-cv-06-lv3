_base_ = [
    # '../_base_/models/custom_fpn_r50.py', 
    '../custom_models/custom_fpn_r50.py',
    # '../_base_/datasets/ade20k.py',
    '../_base_/datasets/handbone_29_classes_640x640.py',  # 커스텀 데이터셋 설정 파일
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (2048, 2048)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor, decode_head=dict(num_classes=29))

val_evaluator = dict(type='DiceCoefficient', num_classes=29)
# test_evaluator = val_evaluator

dataset_type = 'XRayDataset'
data_root = '/data/ephemeral/home/level2-cv-semanticsegmentation-cv-06-lv3/data'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    # dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    # dict(type='Resize', scale=(640,640), keep_ratio=True),
    dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    # dict(type='LoadXRayAnnotations'),
    dict(type='PackSegInputs')
]

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
            seg_map_path=''),
        pipeline=test_pipeline))

test_evaluator = dict(
    type='LREMetric',
    output_dir='work_dirs/format_results'

)

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='WandbVisBackend',
             init_kwargs=dict(
                 project='mmsegmentation',
                 name='fpn_r50_40k_test',
                #  entity='your_wandb_username'
             ),
             save_dir='wandb_logs')
    ],
    name='visualizer'
)
