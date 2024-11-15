import segmentation_models_pytorch as smp
from torch import nn
import torch.nn.functional as F

from src.utils.data_utils import load_yaml_config

class SmpModel(nn.Module):
    def __init__(self, model_config_path: str):
        super(SmpModel, self).__init__()

        # 모델 구성 로드
        self.model_config = load_yaml_config(model_config_path)

        # 모델 생성
        self.model = smp.create_model(
            **self.model_config["model"]
        )

        # 손실 함수 설정
        self.loss_config = self.model_config.get("loss", {})
        self.criterion = self._get_loss_function(self.loss_config)
    
    def _get_loss_function(self, loss_config):
        """
        손실 함수 생성 및 반환.
        loss_config에서 손실 함수의 이름과 파라미터를 받아서 적절한 손실 함수를 반환.
        """
        loss_name = loss_config.get("name", "BCEWithLogitsLoss")  # 기본값은 BCE
        loss_args = loss_config.get("args", {})

        # JaccardLoss나 다른 손실 함수들이 segmentation_models_pytorch에 정의되어 있을 경우
        if loss_name == "JaccardLoss":
            return smp.losses.JaccardLoss(**loss_args)
        elif loss_name == "BCEWithLogitsLoss":
            return smp.losses.BCEWithLogitsLoss(**loss_args)
        elif loss_name == "DiceLoss":
            return smp.losses.DiceLoss(**loss_args)
        elif loss_name == "TverskyLoss":
            return smp.losses.TverskyLoss(**loss_args)
        elif loss_name == "FocalLoss":
            return smp.losses.FocalLoss(**loss_args)
        elif loss_name == "LovaszLoss":
            return smp.losses.LovaszLoss(**loss_args)
        elif loss_name == "SoftBCEWithLogitsLoss":
            return smp.losses.SoftBCEWithLogitsLoss(**loss_args)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def forward(self, images, labels=None):
        # 이미지에 대한 예측 출력
        outputs = self.model(images)

        # 레이블이 주어지면 손실 계산
        if labels is not None:
            # 만약 output과 mask의 사이즈가 맞지 않다면, output을 mask의 사이즈에 맞춰주는 작업을 진행
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = labels.size(-2), labels.size(-1)
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = self.criterion(outputs, labels)
            return outputs, loss
        
        # 레이블이 없으면 예측값만 반환
        return outputs
