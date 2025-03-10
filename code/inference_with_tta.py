import datetime
import json
import os
import pickle
import random
import sys
from functools import partial

import albumentations as A

# external library
import cv2
import numpy as np
import pandas as pd

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# visualization
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm.auto import tqdm

sys.stdout = open('inference_output.txt','w')

# 데이터 경로를 입력하세요

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

BATCH_SIZE = 1
LR = 1e-4
RANDOM_SEED = 21

NUM_EPOCHS = 100
VAL_EVERY = 1

SAVED_DIR = "custom_checkpoint"

IMAGE_ROOT = "../data/test/DCM"
pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}
class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        # image = image.transpose(2, 0, 1)  
        image = image.transpose(2, 0, 1)    # gray_scale
        
        image = torch.from_numpy(image).float()
            
        return image, image_name

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)
# RLE로 인코딩된 결과를 mask map으로 복원합니다.

def get_test_loader(path):
    with open(path,'rb') as g:
        data = pickle.load(g)
    return data

def test_each_label_tta(thr=0.5):
    df_list = {}
    total_df = pd.DataFrame({
            "image_name": [],
            "class": [],
            "rle": [],
        })
    for c in CLASSES:
        torch.cuda.empty_cache()
        print(c, 'model predict')
        data_loader = get_test_loader(os.path.join(SAVED_DIR, f'{c}/fcn_resnet50_best_model_loader.pkl'))
        model = torch.load(os.path.join(SAVED_DIR, f"{c}/fcn_resnet50_best_model.pt"),weights_only=False)
        model = model.cuda()
        model.eval()

        rles = []
        filename_and_class = []
        with torch.no_grad():
            n_class = len(CLASSES)
            for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
                print(images.shape)
                images = images.cuda()    
                outputs = model(images)['out']
                
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")

                # Flip 이미지 예측
                flipped_images = torch.flip(images, dims=[-1])  # 가로축 Flip
                flipped_outputs = model(flipped_images)['out']

                # Flip 예측 결과를 원래 방향으로 다시 Flip
                flipped_outputs = torch.flip(flipped_outputs, dims=[-1])
                flipped_outputs = F.interpolate(flipped_outputs, size=(2048, 2048), mode="bilinear")
                
                # # 평균 후 시그모이드 thr 적용
                combined_outputs = (torch.sigmoid((outputs + flipped_outputs)/2) > thr).detach().cpu().numpy()

                for output, image_name in zip(combined_outputs, image_names):
                    for idx, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{IND2CLASS[idx]}_{image_name}")

        classes, filename = zip(*[x.split("_") for x in filename_and_class])
        image_name = [os.path.basename(f) for f in filename]
        df = pd.DataFrame({
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        })
        df_list[c] = df
        total_df = pd.concat([total_df,df[df["class"]==c]])
        df.to_csv(f"./checkpoints/{c}/output.csv", index=False)

    df_list['total'] = total_df.sort_index()
    total_df.sort_index().to_csv("output.csv", index=False)
    torch.cuda.empty_cache()
    return df_list

df_list = test_each_label_tta()