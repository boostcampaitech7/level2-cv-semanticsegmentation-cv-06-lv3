# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader

def convert_beit_full_model(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        # 백본 변환
        if k.startswith('patch_embed'):
            new_key = k.replace('patch_embed.proj', 'patch_embed.projection')
            new_ckpt[new_key] = v
        elif k.startswith('blocks'):
            new_key = k.replace('blocks', 'layers')
            if 'norm' in new_key:
                new_key = new_key.replace('norm', 'ln')
            elif 'mlp.fc1' in new_key:
                new_key = new_key.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in new_key:
                new_key = new_key.replace('mlp.fc2', 'ffn.layers.1')
            new_ckpt[new_key] = v

        # neck 변환
        elif k.startswith('neck'):
            new_key = k.replace('neck', 'feature2pyramid')
            new_ckpt[new_key] = v

        # decode_head 변환
        elif k.startswith('decode_head'):
            new_key = k.replace('decode_head', 'decode_head')
            if 'norm' in new_key:
                new_key = new_key.replace('norm', 'ln')
            new_ckpt[new_key] = v

        # auxiliary_head 변환
        elif k.startswith('auxiliary_head'):
            new_key = k.replace('auxiliary_head', 'auxiliary_head')
            if 'norm' in new_key:
                new_key = new_key.replace('norm', 'ln')
            new_ckpt[new_key] = v

        # 나머지 키들은 그대로 추가
        else:
            new_key = k
            new_ckpt[new_key] = v

    return new_ckpt

def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained full BEiT models to'
                    'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_beit_full_model(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
