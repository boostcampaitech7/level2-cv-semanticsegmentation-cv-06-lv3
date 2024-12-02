# python native
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# visualization
import matplotlib.pyplot as plt

SAVED_DIR = "checkpoints"

# 테스트 데이터 경로를 입력하세요
IMAGE_ROOT = "/level2-cv-semanticsegmentation-cv-06-lv3/data/test/DCM"
pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

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

# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.

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
        image = image.transpose(2, 0, 1)  
        
        image = torch.from_numpy(image).float()
            
        return image, image_name
    
def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)['out']
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class

tf = A.Resize(512, 512)
test_dataset = XRayInferenceDataset(transforms=tf)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

def run_fold_inference(n_splits=5):
    """
    각 폴드 모델로 추론을 수행하고 결과를 개별 CSV 파일로 저장합니다.
    """
    for fold_idx in range(n_splits):
        print(f'\n[Fold {fold_idx + 1}/{n_splits}] 모델을 사용하여 추론을 시작합니다.')
        
        # 폴드별 모델 로드
        model_path = os.path.join(SAVED_DIR, f"fcn_resnet50_best_model_fold{fold_idx + 1}.pt")
        model = torch.load(model_path)
        
        # 폴드별 추론
        rles, filename_and_class = test(model, test_loader)
        print(filename_and_class[0])
        
        # 추론 결과를 저장
        classes, filenames = zip(*[x.split("_") for x in filename_and_class])
        image_names = [os.path.basename(f) for f in filenames]
        
        df = pd.DataFrame({
            "image_name": image_names,
            "class": classes,
            "rle": rles,
        })

        print(df.head(30))
        os.makedirs('k_fold_output', exist_ok=True)
        output_file = f"k_fold_output/output_fold{fold_idx + 1}.csv"
        df.to_csv(output_file, index=False)
        print(f"[Fold {fold_idx + 1}] 추론 결과 저장 완료: {output_file}")

run_fold_inference(n_splits=5)