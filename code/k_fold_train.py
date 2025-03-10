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

DEVICE = 'cuda:1'

# 데이터 경로를 입력하세요
IMAGE_ROOT = "/level2-cv-semanticsegmentation-cv-06-lv3/data/train/DCM"
LABEL_ROOT = "/level2-cv-semanticsegmentation-cv-06-lv3/data/train/outputs_json"

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

TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
K_SPLITS = 5

LR = 1e-4
RANDOM_SEED = 21

NUM_EPOCHS = 100
VAL_EVERY = 10

SAVED_DIR = "checkpoints"

if not os.path.exists(SAVED_DIR):                                                           
    os.makedirs(SAVED_DIR)

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}

jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

pngs = sorted(pngs)
jsons = sorted(jsons)

class XRayDataset(Dataset):
    def __init__(self, filenames, labelnames, is_train=True, transforms=None):
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # channel first 포맷으로 변경합니다.
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label
    
# 시각화를 위한 팔레트를 설정합니다.
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

# 시각화 함수입니다. 클래스가 2개 이상인 픽셀을 고려하지는 않습니다.
def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
        
    return image

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def save_model(model, file_name='fcn_resnet50_best_model.pt'):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)
    print(f"{output_path} 모델 저장 완료! ")

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            model = model.to(DEVICE)
            
            outputs = model(images)['out']
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice

def train(model, data_loader, val_loader, criterion, optimizer):
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.
    
    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            model = model.to(DEVICE)
            
            outputs = model(images)['out']
            
            # loss를 계산합니다.
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
             
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)

def get_fold_data(fold_idx, gkf_split):
    """
    K-Fold의 특정 폴드 번호(fold_idx)에 해당하는 학습 및 검증 데이터를 반환합니다.
    """

    train_filenames, train_labelnames = [], []
    valid_filenames, valid_labelnames = [], []

    for i, (train_idx, valid_idx) in enumerate(gkf_split):
        if i == fold_idx:
            valid_filenames = [pngs[idx] for idx in valid_idx]
            valid_labelnames = [jsons[idx] for idx in valid_idx]
        else:
            train_filenames.extend([pngs[idx] for idx in train_idx])
            train_labelnames.extend([jsons[idx] for idx in train_idx])

    return train_filenames, train_labelnames, valid_filenames, valid_labelnames

def save_fold_data(fold_idx, train_filenames, train_labelnames, valid_filenames, valid_labelnames):
    """
    각 폴드의 학습 및 검증 데이터 파일 목록을 JSON 파일로 저장하고, 
    라벨 정보는 별도의 JSON 파일로 저장합니다.
    """
    # 파일 목록 JSON 저장
    file_data = {
        "train": train_filenames,
        "valid": valid_filenames
    }

    os.makedirs('k_fold_data', exist_ok=True)
    
    with open(f'k_fold_data/fold_{fold_idx + 1}_files.json', 'w') as f:
        json.dump(file_data, f, indent=4)
    print(f"[Fold {fold_idx + 1}] 파일 목록 저장 완료: fold_{fold_idx + 1}_files.json")

    # 라벨 정보 JSON 저장
    label_data = {
        "train_labels": train_labelnames,
        "valid_labels": valid_labelnames
    }
    
    with open(f'k_fold_data/fold_{fold_idx + 1}_labels.json', 'w') as f:
        json.dump(label_data, f, indent=4)
    print(f"[Fold {fold_idx + 1}] 라벨 정보 저장 완료: fold_{fold_idx + 1}_labels.json")

def run_kfold_training(n_splits=5):
    """
    K-Fold로 모델 학습을 반복하고, 각 폴드 별 최적의 모델을 저장합니다.
    """
    tf = A.Resize(512, 512)
    groups = [os.path.dirname(fname) for fname in pngs]
    gkf = GroupKFold(n_splits=n_splits)
    gkf_split = list(gkf.split(pngs, [0] * len(pngs), groups))  # 제너레이터를 리스트로 변환

    for fold_idx in range(n_splits):
        print(f'\n[Fold {fold_idx + 1}/{n_splits}] 시작합니다.')
        
        # K-Fold 데이터 분할
        train_filenames, train_labelnames, valid_filenames, valid_labelnames = get_fold_data(fold_idx, gkf_split)
        save_fold_data(fold_idx, train_filenames, train_labelnames, valid_filenames, valid_labelnames)
        
        # Dataset 및 DataLoader 설정
        train_dataset = XRayDataset(train_filenames, train_labelnames, is_train=True, transforms=tf)
        valid_dataset = XRayDataset(valid_filenames, valid_labelnames, is_train=False, transforms=tf)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
        
        # 모델 초기화
        model = models.segmentation.fcn_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
        model = model.to(DEVICE)

        # 손실 함수 및 옵티마이저 설정
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)
        
        # 시드 설정
        set_seed()
        
        # 학습 시작
        print(f'\n[Fold {fold_idx + 1}/{n_splits}] 학습을 시작합니다.')
        train(model, train_loader, valid_loader, criterion, optimizer)


run_kfold_training(n_splits=K_SPLITS)