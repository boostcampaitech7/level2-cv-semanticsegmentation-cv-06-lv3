# custom_metrics.py

import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from typing import List, Sequence, Optional
import torch.nn.functional as F

# 클래스 이름을 정의
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

def dice_coef(y_true, y_pred):
    """
    Dice 계수를 계산합니다.

    Args:
        y_true (torch.Tensor): 실제 라벨 텐서.
        y_pred (torch.Tensor): 예측된 라벨 텐서.

    Returns:
        torch.Tensor: 클래스별 Dice 계수.
    """
    y_true_f = y_true.flatten(2)  # 형태: [배치, 클래스, H*W]
    y_pred_f = y_pred.flatten(2)  # 형태: [배치, 클래스, H*W]
    intersection = torch.sum(y_true_f * y_pred_f, dim=-1)  # 형태: [배치, 클래스]

    eps = 1e-4
    return (2. * intersection + eps) / (torch.sum(y_true_f, dim=-1) + torch.sum(y_pred_f, dim=-1) + eps)

@METRICS.register_module()
class DiceCoefficient(BaseMetric):
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        """
        Args:
            num_classes (int): 클래스의 수.
            ignore_index (int, optional): 무시할 인덱스. 기본값은 None.
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """내부 평가 상태를 초기화합니다."""
        self.dices = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        데이터 배치를 처리하고 예측 및 타겟을 저장합니다.

        Args:
            data_batch (dict): 모델 출력이 포함된 데이터 배치.
            data_samples (Sequence[dict]): 실제 타겟이 포함된 데이터 샘플 시퀀스.
        """
        # import pdb
        # pdb.set_trace()
        # data_samples에서 예측값(seg_logits)을 추출하여 스택합니다.
        preds = torch.stack([sample['seg_logits']['data'] for sample in data_samples])  # 형태: [배치, 클래스, H, W]

        # data_samples에서 실제 타겟(gt_sem_seg)을 추출하여 스택합니다.
        targets = torch.stack([sample['gt_sem_seg']['data'] for sample in data_samples])  # 형태: [배치, H, W]

        # 예측과 타겟의 공간적 크기가 다른 경우, 예측을 타겟 크기로 보간합니다.
        output_h, output_w = preds.size(-2), preds.size(-1)
        mask_h, mask_w = targets.size(-2), targets.size(-1)

        if output_h != mask_h or output_w != mask_w:
            preds = F.interpolate(preds, size=(mask_h, mask_w), mode="bilinear", align_corners=False)

        # 시그모이드 활성화 함수 적용 후 임계값을 통해 이진화
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float().detach().cpu()  # 형태: [배치, 클래스, H, W]
        targets = targets.detach().cpu()

        # Dice 계수 계산
        dice = dice_coef(targets, preds)  # 형태: [배치, 클래스]
        self.dices.append(dice)

    def compute_metrics(self, results: List[dict]) -> dict:
        """
        각 클래스에 대한 Dice 계수를 계산합니다.

        Args:
            results (List[dict]): 예측과 타겟이 포함된 결과 리스트.

        Returns:
            dict: 클래스별 Dice 계수와 평균 Dice 계수를 포함한 딕셔너리.
        """
        # 처리된 모든 Dice 점수를 연결
        dices = torch.cat(self.dices, dim=0)  # 형태: [전체 배치, 클래스]
        dices_per_class = torch.mean(dices, dim=0)  # 형태: [클래스]

        # Dice 점수를 클래스 이름과 매핑
        dice_str = [
            f"{c:<12}: {d.item():.4f}"
            for c, d in zip(CLASSES, dices_per_class)
        ]
        dice_str = "\n".join(dice_str)
        print("클래스별 Dice 계수:\n" + dice_str)

        # 모든 클래스에 대한 평균 Dice 점수 계산
        avg_dice = torch.mean(dices_per_class).item()

        # 클래스 이름을 키로 하고 Dice 점수를 값으로 하는 딕셔너리 생성
        class_dice = {f'{c}': d.item() for c, d in zip(CLASSES, dices_per_class)}
        class_dice['mean_dice'] = avg_dice

        return class_dice

    def evaluate(self, size: int) -> dict:
        """
        최종 메트릭을 집계하고 계산합니다.

        Args:
            size (int): 전체 샘플 수.

        Returns:
            dict: 최종 평가 메트릭.
        """
        return self.compute_metrics(None)
