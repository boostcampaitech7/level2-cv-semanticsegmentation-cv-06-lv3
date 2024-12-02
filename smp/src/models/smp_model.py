import segmentation_models_pytorch as smp
from torch import nn
import torch.nn.functional as F

from src.utils.data_utils import load_yaml_config

class SmpModel(nn.Module):
    def __init__(self, config):
        super(SmpModel, self).__init__()

        # load config files
        self.model_config = load_yaml_config(config["path"].get("model", ""))
        self.loss_config = load_yaml_config(config["path"].get("loss", ""))
        self.experiments_config = load_yaml_config(config["path"].get("experiments", ""))

        # create model (segementation models with pytorch)
        self.model = smp.create_model(
            **self.model_config["model"]
        )

        if "interpolation" in self.experiments_config and self.experiments_config["interpolation"].get("enabled", False):
            self.mode = self.experiments_config["interpolation"].get("mode", "bilinear")  # 기본값 bilinear
        else:
            self.mode = "bilinear"  # interpolation이 비활성화된 경우 기본값

        # 손실 함수 설정
        self.loss_config = self.loss_config.get("loss", {})
        self.criterion = self._get_loss_function(self.loss_config)
    
    def _get_loss_function(self, loss_config):
        """
        손실 함수 생성 및 반환.
        loss_config에서 손실 함수의 이름과 파라미터를 받아 적절한 손실 함수를 반환.
        여러 손실 함수를 조합하여 사용하는 경우도 처리.
        """
        use_different_loss = loss_config.get("use_different_loss", False)
        
        if not use_different_loss:  # 단일 손실 함수 사용
            loss_name = loss_config.get("name", "BCEWithLogitsLoss")  # 기본값
            loss_args = loss_config.get("args", {})
            return self._create_single_loss(loss_name, loss_args)
        
        # 여러 손실 함수 조합
        components = loss_config.get("components", [])
        if not components:
            raise ValueError("`components`가 비어 있습니다. 손실 함수를 하나 이상 정의하세요.")
        
        loss_functions = []
        weights = []
        
        for component in components:
            loss_name = component.get("name")
            loss_args = component.get("args", {})
            weight = component.get("weight", 1.0)  # 기본 가중치
            
            loss_fn = self._create_single_loss(loss_name, loss_args)
            loss_functions.append((loss_fn, weight))
        
        # 조합된 손실 함수 반환
        return self._combine_loss_functions(loss_functions)

    def _create_single_loss(self, loss_name, loss_args):
        """
        단일 손실 함수 생성.
        """
        if loss_name == "JaccardLoss":
            return smp.losses.JaccardLoss(**loss_args)
        elif loss_name == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss(**loss_args)
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

    def _combine_loss_functions(self, loss_functions):
        """
        여러 손실 함수를 조합하여 단일 손실 함수로 반환.
        """
        def combined_loss(y_pred, y_true):
            total_loss = 0.0
            for loss_fn, weight in loss_functions:
                total_loss += weight * loss_fn(y_pred, y_true)
            return total_loss
        
        return combined_loss

    
    def forward(self, images, labels=None):
        # 이미지에 대한 예측 출력
        outputs = self.model(images)

        # 레이블이 주어지면 손실 계산
        if labels is not None:
            # 만약 output과 mask의 사이즈가 맞지 않다면, output을 mask의 사이즈에 맞춰주는 작업을 진행
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = labels.size(-2), labels.size(-1)
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode=self.mode)
            
            loss = self.criterion(outputs, labels)
            return outputs, loss
        
        # 레이블이 없으면 예측값만 반환
        return outputs
