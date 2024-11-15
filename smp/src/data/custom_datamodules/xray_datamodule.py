from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.data.base_datamodule import BaseDataModule
from src.data.datasets.xray_dataset import XRayDataset
from src.data.datasets.xray_inferencedataset import XRayInferenceDataset
from src.utils.data_utils import load_yaml_config 
from src.utils.seed_utils import set_seed


class XRayDataModule(BaseDataModule):
    def __init__(self, data_config_path: str, augmentation_config_path: str, seed: int=21):
        self.data_config = load_yaml_config(data_config_path)
        self.augmentation_config = load_yaml_config(augmentation_config_path)
        self.seed = self.data_config['seed']
        super().__init__(self.data_config)

    def setup(self, stage: Optional[str] = None):
        # set seed
        set_seed(self.seed)

        # load datasets
        if self.augmentation_config["augmentation"]["use_augmentation"]:
            train_transforms = self._get_augmentation_transforms()

        else:
            train_transforms = A.Compose(
                [A.Resize(512, 512)],
            )

        test_transforms = A.Compose(
            [A.Resize(512, 512)],
        )

        self.collate_fn = None

        train_data_path = self.config["data"]["train_data_path"]
        train_label_path = self.config["data"]["train_label_path"]
        test_data_path = self.config["data"]["test_data_path"]

        self.train_dataset = XRayDataset(
            image_path= train_data_path,
            label_path= train_label_path,
            is_train=True,
            transforms=train_transforms
        )
        
        self.val_dataset = XRayDataset(
            image_path= train_data_path,
            label_path= train_label_path,
            is_train=False,
            transforms=test_transforms
        )

        self.test_dataset = XRayInferenceDataset(
            image_path= test_data_path,
            transforms=test_transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["train"]["batch_size"],
            num_workers=self.config["data"]["train"]["num_workers"],
            shuffle=True,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["val"]["batch_size"],
            num_workers=self.config["data"]["val"]["num_workers"],
            shuffle=False,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["test"]["batch_size"],
            num_workers=self.config["data"]["test"]["num_workers"],
            shuffle=False,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )
    
    def _get_augmentation_transforms(self):
        transform_list = []
        for transform_config in self.augmentation_config["augmentation"]["transforms"]:
            transform_name = transform_config["name"]
            
            # TODO Resize를 RandomCrop과 병행할지 결정해보기
            if transform_name == "OneOf":
                # OneOf 변환을 따로 처리
                sub_transforms = []
                for sub_transform in transform_config["params"]["transforms"]:
                    transform_class = getattr(A, sub_transform["name"])
                    sub_transforms.append(transform_class(**sub_transform["params"]))


                transform_list.append(A.OneOf(
                    sub_transforms,
                    p=transform_config["params"]["p"]
                    ))

                
                
                
            elif transform_name == "SomeOf":
                # OneOf 변환을 따로 처리
                sub_transforms = []
                for sub_transform in transform_config["params"]["transforms"]:
                    transform_class = getattr(A, sub_transform["name"])
                    sub_transforms.append(transform_class(**sub_transform["params"]))
                
                # OneOf 변환 추가
                transform_list.append(A.SomeOf(
                    sub_transforms, 
                    p=transform_config["params"]["p"],
                    n=transform_config["params"]["n"],
                    ))
                
            else:
                transform_class = getattr(A, transform_name)
                transform_list.append(transform_class(**transform_config["params"]))
        
        return A.Compose(
            transform_list,
            )
        