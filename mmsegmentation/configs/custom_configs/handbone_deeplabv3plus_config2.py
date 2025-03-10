_base_ = [
    '../custom_models/custom_deeplabv3plus.py', 
    # '../_base_/datasets/ade20k.py',
    '../_base_/datasets/handbone_29_classes_1024x1024.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_25k.py'
]
crop_size = (2048, 2048)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=29),
    auxiliary_head=dict(num_classes=29))
