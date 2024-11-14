# custom_metrics.py

import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from typing import List, Sequence, Optional
import torch.nn.functional as F


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)


@METRICS.register_module()
class DiceCoefficient(BaseMetric):
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        """
        Args:
            num_classes (int): Number of classes.
            ignore_index (int, optional): Index to ignore. Defaults to None.
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Resets the internal evaluation state."""
        # self.preds = []
        # self.targets = []
        self.dices = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Processes a batch of data and stores predictions and targets.

        Args:
            data_batch (dict): A batch of data containing model outputs.
            data_samples (Sequence[dict]): A sequence of data samples containing ground truth.
        """
        # import pdb

        # print(data_batch)
        # print(data_samples)
        # pdb.set_trace()
        # preds = data_batch['outputs']  # 모델의 출력 (logits)

        preds = torch.stack([i['seg_logits']['data'] for i in data_samples])
        # targets = data_samples['gt_semantic_seg']  # 실제 마스크

        targets = torch.stack([i.gt_sem_seg.data for i in data_batch['data_samples']])



        output_h, output_w = preds.size(-2), preds.size(-1)
        mask_h, mask_w = targets.size(-2), targets.size(-1)

        if output_h != mask_h or output_w != mask_w:
            preds = F.interpolate(preds, size=(mask_h, mask_w), mode="bilinear")

        # TODO pred 값에 이 부분 있는듯  
        # 예측값에 시그모이드 적용 후 임계값을 통해 이진화
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float().detach().cpu()
        targets = targets.detach().cpu()

        dice = dice_coef(preds, targets)

        self.dices.append(dice)
        
        # ignore_index가 설정된 경우 해당 인덱스 마스크
        # if self.ignore_index is not None:
        #     mask = targets != self.ignore_index
        #     preds = preds * mask.unsqueeze(1)
        #     targets = targets * mask

        # 다중 레이블이므로 각 클래스별로 마스크 분리
        # for cls in range(self.num_classes):
        #     self.preds.append(preds[:, cls, :, :].cpu())
        #     self.targets.append((targets == cls).float().cpu())



    def compute_metrics(self, results: List[dict]) -> dict:
        """
        Computes the Dice coefficient for each class.

        Args:
            results (List[dict]): A list of results containing predictions and targets.

        Returns:
            dict: A dictionary with Dice coefficients for each class and the mean Dice.
        """
        # dice_scores = torch.zeros(self.num_classes)
        # for cls in range(self.num_classes):
        #     pred = self.preds[cls]
        #     target = self.targets[cls]

        #     intersection = (pred * target).sum()
        #     union = pred.sum() + target.sum()

        #     if union == 0:
        #         dice = 1.0  # 두 집합이 모두 비어있다면 Dice는 1
        #     else:
        #         dice = (2. * intersection + 1e-6) / (union + 1e-6)
        #     dice_scores[cls] += dice

        # # 평균 Dice 계산
        # mean_dice = dice_scores.mean().item()

        # # 각 클래스별 Dice
        # class_dice = {f'class_{cls}': dice_scores[cls].item() for cls in range(self.num_classes)}

        dices = torch.cat(self.dices, 0)
        dices_per_class = torch.mean(dices, 0)

        class_dice = {f'class_{cls}': dices_per_class[cls].item() for cls in range(self.num_classes)}

        avg_dice = torch.mean(dices_per_class).item()

        class_dice['mean_dice'] = avg_dice

        return class_dice

    def evaluate(self, size: int) -> dict:
        """
        Aggregates and computes the final metrics.

        Args:
            size (int): Total number of samples.

        Returns:
            dict: Final evaluation metrics.
        """
        return self.compute_metrics(None)
