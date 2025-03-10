import os
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime

import torch.nn as nn

from src.models.smp_model import SmpModel
from src.models.gender_classifier import GenderClassifier
from src.models.leftright_classifier import LeftRightClassifier
from src.utils.data_utils import load_yaml_config
from src.utils.constants import IND2CLASS, GENDER, LEFTRIGHT, THRESHOLD
from src.utils.sliding_window import SlidingWindowInference

class SmpModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = SmpModel(config=config)
        
        self.validation_outputs = []
        self.rles = []
        self.filename_and_class = []

        self._setup()

    #################################################################################

    def _setup(self):
        self._load_configs()
        self._setup_auxclassifier()
        self._setup_interpolation()
        self._setup_sliding_window()
        self._setup_crf()
        self._replace_batchnorm_with_groupnorm()

    def _load_configs(self):
        """Load train and util configs."""
        self.train_config = load_yaml_config(self.config["path"].get("train", ""))
        self.experiments_config = load_yaml_config(self.config["path"].get("experiments", ""))

    def _setup_auxclassifier(self):
        """Set up auxclassifier for multi-task learning"""
        self.gender = self.experiments_config["metadata"]["gender"].get("enabled", False)
        self.leftright = self.experiments_config["metadata"]["leftright"].get("enabled", False)
        
        self.gender_classifier = GenderClassifier() if self.gender else None
        self.leftright_classifier = LeftRightClassifier() if self.leftright else None

    def _setup_interpolation(self):
        """Set up interpolation mode."""
        interpolation_config = self.experiments_config.get("interpolation", {})
        self.mode = interpolation_config.get("mode", "bilinear") if interpolation_config.get("enabled", False) else "bilinear"

    def _setup_sliding_window(self):
        """Set up sliding window parameters."""
        sliding_window_config = self.experiments_config.get("sliding_window", {})
        self.sliding_window = sliding_window_config.get("enabled", False)
        if self.sliding_window:
            self.sliding_window_infer = SlidingWindowInference(
                model=self.model, 
                patch_size=sliding_window_config.get("patch_size", 1024),
                batch_size=sliding_window_config.get("batch_size", 1),
                num_classes=self.experiments_config.get("num_classes", 29)
                )

    def _setup_crf(self):
        """Set up CRF parameters."""
        self.crf = self.experiments_config.get("crf", {}).get("enabled", False)

    def _replace_batchnorm_with_groupnorm(self):
        """Replace BatchNorm layers with GroupNorm layers."""
        self.model = self._convert_batchnorm_to_groupnorm(self.model)

    def _convert_batchnorm_to_groupnorm(self, model, default_num_groups=16):
        """Convert BatchNorm2d layers to GroupNorm."""
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                group_norm = self._create_groupnorm(module.num_features, default_num_groups)
                setattr(model, name, group_norm)
            else:
                self._convert_batchnorm_to_groupnorm(module)
        return model
    
    @staticmethod
    def _create_groupnorm(num_channels, default_num_groups):
        """Create a GroupNorm layer."""
        num_groups = min(default_num_groups, num_channels)
        if num_channels % num_groups != 0:
            num_groups = 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    
    #################################################################################
    
    def forward(self, images, labels=None):
        return self.model(images, labels)
    
    def common_step(self, batch):
        if self.gender or self.leftright:
            images, labels, metas = batch
        else:
            images, labels = batch

        outputs, loss = self.model(images, labels)

        gender_loss = None
        leftright_loss = None

        if self.gender:
            # metas에서 gender 정보를 추출
            genders = [GENDER.get(meta, GENDER["female"]) for meta in metas['gender']]

            # genders를 텐서로 변환 (batch_size 크기)
            genders = torch.tensor(genders, dtype=torch.long, device=outputs.device)

            # gender_classifier 호출 (배치 입력)
            _, gender_loss = self.gender_classifier(outputs, genders)
        
        if self.leftright:
            # metas에서 leftright 정보를 추출
            leftrights = [LEFTRIGHT.get(meta, LEFTRIGHT["왼손"]) for meta in metas['label']]

            # leftrights 텐서로 변환 (batch_size 크기)
            leftrights = torch.tensor(leftrights, dtype=torch.long, device=outputs.device)

            # leftright_classifier 호출 (배치 입력)
            _, leftright_loss = self.leftright_classifier(outputs, leftrights)
        
        return outputs, labels, loss, gender_loss, leftright_loss

    
    def training_step(self, batch, batch_idx):
        _, _, loss, gender_loss, leftright_loss = self.common_step(batch)

        # log
        self.log('train_loss', loss, prog_bar=True)
        self.log('gender_train_loss', gender_loss or 0.0)
        self.log('leftright_train_loss', leftright_loss or 0.0)
        if gender_loss:
            loss += 0.005 * gender_loss
        if leftright_loss:
            loss += 0.005 * leftright_loss

        return loss

    def validation_step(self, batch, batch_idx):
        if not self.sliding_window:
            outputs, labels, loss, gender_loss, leftright_loss = self.common_step(batch)
        else:
            images, labels = batch
            outputs = self.sliding_window_infer._sliding_window_step(images)
            loss = self.model.criterion(outputs, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log('gender_val_loss', gender_loss or 0.0)
        self.log('leftright_val_loss', leftright_loss or 0.0)

        outputs = torch.sigmoid(outputs)
        outputs = (outputs > 0.5).float()
        labels = labels.float()

        dices = self.dice_coef(outputs, labels)
        self.validation_outputs.append(dices)

        return dices

    def test_step(self, batch, batch_idx):
        images, image_names = batch
        if self.sliding_window:
            outputs = self.sliding_window_infer._sliding_window_step(images)
        else:
            outputs = self.model(images)
            outputs = F.interpolate(outputs, size=(2048, 2048), mode=self.mode, align_corners=True)

        outputs = torch.sigmoid(outputs)

        for output, image_name in zip(outputs, image_names):
            self._generate_rles(output, image_name)

    #################################################################################

    def on_validation_epoch_end(self):
        if len(self.validation_outputs) == 0:
            return  # validation_outputs가 비어 있으면 종료

        # 전체 dices를 하나로 합침
        all_dices = torch.cat(self.validation_outputs, dim=0)  

        # 클래스별 Dice score 평균 계산
        dices_per_class = all_dices.mean(dim=0)  

        # 전체 평균 Dice score 계산
        avg_dice = dices_per_class.mean().item()

        # 각 클래스에 대한 Dice score 기록
        for i, dice in enumerate(dices_per_class):
            self.log(f'{IND2CLASS[i]}_dice', dice.item(), on_epoch=True, prog_bar=False)

        # 전체 평균 Dice 기록
        self.log('avg_dice_score', avg_dice, on_epoch=True, prog_bar=True)

        # validation outputs 초기화
        self.validation_outputs = []

        # Return avg dice score for early stopping or logging
        return avg_dice
    
    def on_test_epoch_end(self):
        classes, filename = zip(*[x.split("_") for x in self.filename_and_class])

        image_name = [os.path.basename(f) for f in filename]

        df = pd.DataFrame({
            "image_name": image_name,
            "class": classes,
            "rle": self.rles,
        }) 

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"./logs/{self.config.get('filename', '')}_{current_time}.csv", index=False)

        return df
    
    #################################################################################

    def _convert_str_params_to_float(self, params):
        """주어진 파라미터 내 모든 문자열 값을 float으로 변환."""
        return {
            key: float(value) if isinstance(value, str) else value
            for key, value in params.items()
        }

    def configure_optimizers(self):
        # optimizer 설정 읽기
        lr_backbone = float(self.train_config["optimizer"]["lr_backbone"])
        optimizer_params = self._convert_str_params_to_float(self.train_config["optimizer"]["params"])

        # 파라미터 그룹 정의
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad], "lr": lr_backbone},
        ]

        # Optimizer 생성
        optimizer_class = getattr(torch.optim, self.train_config["optimizer"]["name"])
        optimizer = optimizer_class(param_dicts, **optimizer_params)

        # Scheduler 설정 (존재하는 경우)
        if "scheduler" in self.train_config:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.train_config["scheduler"]["name"])
            scheduler = scheduler_class(optimizer, **self.train_config["scheduler"]["params"])
            return [optimizer], [scheduler]
        
        return optimizer
    
    #################################################################################

    @staticmethod
    def dice_coef(y_true, y_pred, eps=1e-4):
        y_true_f = y_true.flatten(2)
        y_pred_f = y_pred.flatten(2)
        intersection = torch.sum(y_true_f * y_pred_f, -1)

        return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)
    
    @staticmethod
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
    
    def _generate_rles(self, output, image_name):
        """Generate RLEs for each class and append to results."""
        for c, segm in enumerate(output):
            # you can adjust threshold class by class (feat. utils/constants.py)
            binary_mask = (segm > THRESHOLD[c]).detach().cpu().numpy()
            rle = self.encode_mask_to_rle(binary_mask)
            self.rles.append(rle)
            self.filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    @staticmethod
    def encode_mask_to_rle(mask):
        """Encode binary mask to RLE format."""
        pixels = SmpModule._prepare_mask_for_rle(mask)
        runs = SmpModule._compute_run_starts_and_lengths(pixels)
        return ' '.join(map(str, runs))

    @staticmethod
    def _prepare_mask_for_rle(mask):
        """Prepare mask for RLE encoding."""
        pixels = mask.flatten()
        return np.concatenate([[0], pixels, [0]])

    @staticmethod
    def _compute_run_starts_and_lengths(pixels):
        """Compute run starts and lengths for RLE."""
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return runs