from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.data.base_datamodule import BaseDataModule
from src.data.datasets.xray_dataset import XRayDataset
from src.data.datasets.xray_inferencedataset import XRayInferenceDataset
from src.data.datasets.xray_validationdataset import XRayValidationDataset
from src.utils.data_utils import load_yaml_config 
from src.utils.seed_utils import set_seed


class XRayDataModule(BaseDataModule):
    def __init__(self, config):
        self.data_config = load_yaml_config(config["path"].get("data", None))
        self.augmentation_config = load_yaml_config(config["path"].get("augmentation", None))
        super().__init__(config)

    def setup(self, stage: Optional[str] = None):
        # set seed
        set_seed(self.data_config.get("seed", 21))

        # setting image_size & validation fold number
        self.image_size = self.data_config.get("image_size", 512)
        self.val_fold = self.data_config.get("val_fold", 0)
        self.collate_fn = None

        # load train transform
        if self.augmentation_config["augmentation"].get("enabled", False):
            train_transforms = self._get_augmentation_transforms()
        else:
            train_transforms = A.Compose(
                [A.Resize(int(self.image_size), int(self.image_size))],
            )

        # load test transform
        test_transforms = A.Compose(
            [A.Resize(int(self.image_size), int(self.image_size))],
        )

        # get data folder path from configs
        train_data_path = self.data_config["data"].get("train_data_path", "")
        train_label_path = self.data_config["data"].get("train_label_path", "")
        test_data_path = self.data_config["data"].get("test_data_path", "")
        meta_data_path = self.data_config["data"].get("meta_data_path", None)

        # define train dataset
        self.train_dataset = XRayDataset(
            image_path=train_data_path,
            label_path=train_label_path,
            meta_path=meta_data_path,
            is_train=True,
            transforms=train_transforms,
            val_fold=self.val_fold
        )
        
        # define val dataset
        self.val_dataset = XRayDataset(
            image_path=train_data_path,
            label_path=train_label_path,
            meta_path=meta_data_path,
            is_train=False,
            transforms=test_transforms,
            val_fold=self.val_fold
        )

        # define test dataset
        self.test_dataset = XRayInferenceDataset(
            image_path=test_data_path,
            transforms=test_transforms
        )

        # define inference dataset (for validation set inference)
        self.inference_dataset = XRayValidationDataset(
            image_path=train_data_path,
            label_path=train_label_path,
            is_train=False,
            transforms=test_transforms,
            val_fold=self.val_fold
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config["data"]["train"].get("batch_size", 2),
            num_workers=self.data_config["data"]["train"].get("num_workers", 4),
            shuffle=True,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config["data"]["val"].get("batch_size", 2),
            num_workers=self.data_config["data"]["val"].get("num_workers", 4),
            shuffle=False,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config["data"]["test"].get("batch_size", 2),
            num_workers=self.data_config["data"]["test"].get("num_workers", 4),
            shuffle=False,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )
    
    def _get_augmentation_transforms(self):
        transform_list = []
        for transform_config in self.augmentation_config["augmentation"]["transforms"]:
            transform_name = transform_config["name"]
            
            # OneOf
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
                
            # SomeOf
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

            

        