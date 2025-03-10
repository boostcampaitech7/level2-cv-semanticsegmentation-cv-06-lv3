# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_vit(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('head'):
            # 기존 스크립트는 'head'로 시작하는 키를 무시
            # 이제 전체 모델을 변환하므로 이 부분을 주석 처리하거나 수정
            # continue
            pass  # 'head' 키도 변환할 수 있도록 수정

        if k.startswith('backbone.'):
            # Backbone의 키 변환
            backbone_k = k[len('backbone.'):]  # 'backbone.' 접두어 제거
            if backbone_k.startswith('norm'):
                new_k = backbone_k.replace('norm.', 'ln1.')
            elif backbone_k.startswith('patch_embed'):
                if 'proj' in backbone_k:
                    new_k = backbone_k.replace('proj', 'projection')
                else:
                    new_k = backbone_k
            elif backbone_k.startswith('blocks'):
                if 'norm' in backbone_k:
                    new_k = backbone_k.replace('norm', 'ln')
                elif 'mlp.fc1' in backbone_k:
                    new_k = backbone_k.replace('mlp.fc1', 'ffn.layers.0.0')
                elif 'mlp.fc2' in backbone_k:
                    new_k = backbone_k.replace('mlp.fc2', 'ffn.layers.1')
                elif 'attn.qkv' in backbone_k:
                    new_k = backbone_k.replace('attn.qkv.', 'attn.attn.in_proj_')
                elif 'attn.proj' in backbone_k:
                    new_k = backbone_k.replace('attn.proj', 'attn.attn.out_proj')
                else:
                    new_k = backbone_k
                new_k = new_k.replace('blocks.', 'layers.')
            else:
                new_k = backbone_k
            new_ckpt[f'backbone.{new_k}'] = v
        elif k.startswith('decode_head.'):
            # Decode Head의 키 변환
            decode_k = k[len('decode_head.'):]  # 'decode_head.' 접두어 제거
            # 여기서 필요한 키 변환 로직을 추가
            # 예: 'attn.'을 'decode_head.attn.'으로 변경 등
            # 예시로 일부 변환을 추가합니다. 실제 키 구조에 맞게 수정 필요
            if decode_k.startswith('norm'):
                new_k = decode_k.replace('norm.', 'ln1.')
            elif decode_k.startswith('mlp.fc1'):
                new_k = decode_k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif decode_k.startswith('mlp.fc2'):
                new_k = decode_k.replace('mlp.fc2', 'ffn.layers.1')
            elif decode_k.startswith('attn.qkv'):
                new_k = decode_k.replace('attn.qkv.', 'attn.attn.in_proj_')
            elif decode_k.startswith('attn.proj'):
                new_k = decode_k.replace('attn.proj', 'attn.attn.out_proj')
            else:
                new_k = decode_k
            # 필요한 추가 변환이 있다면 여기에 추가
            new_ckpt[f'decode_head.{new_k}'] = v
        else:
            # 기타 키는 그대로 유지
            new_ckpt[k] = v

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
                    'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_vit(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
