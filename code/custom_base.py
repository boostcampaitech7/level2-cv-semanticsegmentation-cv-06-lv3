# python native
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

sys.stdout = open('custom_base_output.txt','w')

# 데이터 경로를 입력하세요

IMAGE_ROOT = "../data/train/DCM"
LABEL_ROOT = "../data/train/outputs_json"

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
    def __init__(self, is_train=True, transforms=None):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
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
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # gray_scale
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
            if c not in CLASS2IND: continue
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
        #image = image[np.newaxis, ...]    # gray_scale
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label
    
tf = A.Resize(2048, 2048)
train_dataset = XRayDataset(is_train=True, transforms=tf)
valid_dataset = XRayDataset(is_train=False, transforms=tf)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    drop_last=True,
)

# 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

TEST_IMAGE_ROOT = "../data/test/DCM"
test_pngs = {
    os.path.relpath(os.path.join(root, fname), start=TEST_IMAGE_ROOT)
    for root, _dirs, files in os.walk(TEST_IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = test_pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(TEST_IMAGE_ROOT, image_name)
        
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

test_dataset = XRayInferenceDataset(transforms=tf)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=4,
    shuffle=False,
    num_workers=0,
    drop_last=False
)

def save_test_loader(path):
    with open(path,'wb') as s:
        pickle.dump(test_loader, s)

def save_score(score, path):
    with open(path,'wb') as s:
        pickle.dump(score, s)

def get_test_loader(path):
    with open(path,'rb') as g:
        data = pickle.load(g)
    return data

def get_score(path):
    with open(path,'rb') as g:
        data = pickle.load(g)
    return data

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def save_model(model, file_name='fcn_resnet50_best_model.pt'):
    output_path = os.path.join(SAVED_DIR, file_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model, output_path)

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def validation_each_class(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
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
            outputs = (outputs > thr)
            
            dice = dice_coef(outputs, masks)
            dices.append(dice.detach().cpu())
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    
    return dices_per_class

def train_save_each(model, data_loader, val_loader, criterion, optimizer):
    print(f'Start training..')
    set_seed()
    n_class = len(CLASSES)
    best_dice = {c:float(get_score(os.path.join(SAVED_DIR, f'{c}/score.pkl')))
                if os.path.isfile(os.path.join(SAVED_DIR, f'{c}/score.pkl'))
                else 0
                for c in CLASSES}

    for epoch in range(NUM_EPOCHS):
        model.train()
        with tqdm(enumerate(data_loader), total=len(data_loader)) as tbar:
            for step, (images, masks) in tbar:            
                # gpu 연산을 위해 device 할당합니다.

                images, masks = images.cuda(), masks.cuda()
                model = model.cuda()

                outputs = model(images)['out']

                # loss를 계산합니다.
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tbar.set_postfix(loss=round(loss.item(),4))
                # step 주기에 따라 loss를 출력합니다.
                if (step + 1) % (640//BATCH_SIZE) == 0:
                    print(
                        f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                        f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                        f'Step [{step+1}/{len(train_loader)}], '
                        f'Loss: {round(loss.item(),4)}'
                    )
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % VAL_EVERY == 0:
            torch.cuda.empty_cache()
            dices_per_class = validation_each_class(epoch + 1, model, val_loader, criterion)
            torch.cuda.empty_cache()
            dice_str = []
            for c, dice in zip(CLASSES, dices_per_class):
                if best_dice[c] < dice.item():
                    dice_str.append(f"{c:<12}: best-{best_dice[c]:.4f}, now-{dice.item():.4f} best dice!!")
                    best_dice[c] = dice.item()
                    save_model(model, file_name=f'{c}/fcn_resnet50_best_model.pt')
                    save_test_loader(os.path.join(SAVED_DIR, f'{c}/fcn_resnet50_best_model_loader.pkl'))
                    save_score(dice.item(),os.path.join(SAVED_DIR, f'{c}/score.pkl'))
                else:
                    dice_str.append(f"{c:<12}: best-{best_dice[c]:.4f}, now-{dice.item():.4f}")
                
            dice_str = "\n".join(dice_str)
            print(dice_str)
            print(f"epoch mean dice: {torch.mean(dices_per_class).item():.4f}")
            print(f"mean of best dice: {sum(best_dice.values())/len(best_dice):.4f}")

model = models.segmentation.fcn_resnet50(pretrained=True)

def replace_batchnorm_with_groupnorm(model, num_groups=32):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # GroupNorm으로 대체
            num_channels = module.num_features
            setattr(model, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        else:
            # 재귀적으로 하위 모듈 처리
            replace_batchnorm_with_groupnorm(module, num_groups)
    return model

model = replace_batchnorm_with_groupnorm(model, num_groups=32)
# output class 개수를 dataset에 맞도록 수정합니다.
model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

# Loss function을 정의합니다.
criterion = nn.BCEWithLogitsLoss()

# Optimizer를 정의합니다.
optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)

# 시드를 설정합니다.
set_seed()

torch.cuda.empty_cache()

train_save_each(model, train_loader, valid_loader, criterion, optimizer)

print()
print('predict start')
print()

IMAGE_ROOT = "../data/test/DCM"
pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

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

def test_each_label(thr=0.5):
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
                images = images.cuda()    
                outputs = model(images)['out']
                
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > thr).detach().cpu().numpy()
                
                for output, image_name in zip(outputs, image_names):
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

torch.cuda.empty_cache()

df_list = test_each_label()