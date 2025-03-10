import os
import json
import numpy as np

import cv2
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

class XRayDataset(Dataset):
    def __init__(self, image_path, label_path, meta_path=None, is_train=True, transforms=None, val_fold=0):

        self.image_path = image_path
        self.label_path = label_path
        self.meta_path = meta_path

        self.CLASSES = [
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]

        self.CLASS2IND = {v: i for i, v in enumerate(self.CLASSES)}
        self.IND2CLASS = {v: k for k, v in self.CLASS2IND.items()}

        # Load meta information
        if self.meta_path:
            with open(self.meta_path, "r") as f:
                meta_info = json.load(f)
                self.meta_info = {meta["image_name"]: meta for meta in meta_info}
        else:
            self.meta_info = None

        pngs = {
            os.path.relpath(os.path.join(root, fname), start=self.image_path)
            for root, _dirs, files in os.walk(self.image_path)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }

        jsons = {
            os.path.relpath(os.path.join(root, fname), start=self.label_path)
            for root, _dirs, files in os.walk(self.label_path)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }

        # 이름 순으로 정렬하여 짝이 맞도록 매칭
        self.pngs = sorted(pngs)
        self.jsons = sorted(jsons)

        _filenames = np.array(self.pngs)
        _labelnames = np.array(self.jsons)

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
                if i == val_fold:
                    continue

                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])

            else:
                if i == val_fold:
                    filenames = list(_filenames[y])
                    labelnames = list(_labelnames[y])
                    break
                else:
                    continue

        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_path, image_name)

        image = cv2.imread(image_path)
        image = image / 255.

        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_path, label_name)

        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(self.CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = self.CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        # 메타 정보 추가
        meta_info = None
        if self.meta_info:
            # image_path에서 ID 추출
            meta_info = self.meta_info.get(os.path.basename(image_name), None)

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

        # 메타 정보 반환
        if self.meta_info:
            return image, label, meta_info
        else:
            return image, label