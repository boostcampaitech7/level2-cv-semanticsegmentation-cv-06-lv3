_base_ = ['./handbone_segformer_test_config.py']

model = dict(
    backbone=dict(
        embed_dims=64,
        # num_layers=[3, 6, 40, 3]), #mit-b5
        # num_layers=[3, 4, 6, 3]  # mit-b2 설정
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512]  # Decode head 입력 채널
    )
)