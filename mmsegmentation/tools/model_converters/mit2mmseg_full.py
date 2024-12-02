# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_mit_full(ckpt):
    new_ckpt = OrderedDict()
    
    # Iterate through each key-value pair in the checkpoint
    for k, v in ckpt.items():
        # Ignore classification head weights if present
        if k.startswith('head'):
            continue

        # Patch embedding conversion for backbone layers
        elif k.startswith('patch_embed'):
            stage_i = int(k.split('.')[0].replace('patch_embed', ''))
            new_k = k.replace(f'patch_embed{stage_i}', f'layers.{stage_i-1}.0')
            new_v = v
            if 'proj.' in new_k:
                new_k = new_k.replace('proj.', 'projection.')

        # Transformer encoder layer conversion for backbone layers
        elif k.startswith('block'):
            stage_i = int(k.split('.')[0].replace('block', ''))
            new_k = k.replace(f'block{stage_i}', f'layers.{stage_i-1}.1')
            new_v = v
            if 'attn.q.' in new_k:
                sub_item_k = k.replace('q.', 'kv.')
                new_k = new_k.replace('q.', 'attn.in_proj_')
                new_v = torch.cat([v, ckpt[sub_item_k]], dim=0)
            elif 'attn.kv.' in new_k:
                continue
            elif 'attn.proj.' in new_k:
                new_k = new_k.replace('proj.', 'attn.out_proj.')
            elif 'mlp.' in new_k:
                new_k = new_k.replace('mlp.', 'ffn.layers.')
                if 'fc1.weight' in new_k or 'fc2.weight' in new_k:
                    new_v = v.reshape((*v.shape, 1, 1))
                new_k = new_k.replace('fc1.', '0.')
                new_k = new_k.replace('fc2.', '4.')

        # Norm layer conversion for backbone layers
        elif k.startswith('norm'):
            stage_i = int(k.split('.')[0].replace('norm', ''))
            new_k = k.replace(f'norm{stage_i}', f'layers.{stage_i-1}.2')
            new_v = v

        # Decode head layer conversion
        elif k.startswith('decode_head'):
            new_k = k.replace('decode_head', 'decode_head')
            new_v = v

        else:
            # Retain all other keys as they are (for additional layers or specific configurations)
            new_k = k
            new_v = v

        new_ckpt[new_k] = new_v
    
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained segformer to '
        'MMSegmentation style.')
    parser.add_argument('src', help='Path to the source model checkpoint')
    parser.add_argument('dst', help='Path to save the converted checkpoint')
    args = parser.parse_args()

    # Load the checkpoint using CheckpointLoader from mmengine
    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Convert checkpoint weights
    weight = convert_mit_full(state_dict)

    # Save converted weights to the destination path
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
