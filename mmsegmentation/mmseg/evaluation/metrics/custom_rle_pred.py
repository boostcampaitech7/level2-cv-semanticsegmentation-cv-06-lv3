# custom_metrics.py

import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from typing import List, Sequence, Optional
import torch.nn.functional as F

import numpy as np
import os 
import pandas as pd 

# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.

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





@METRICS.register_module()
class LREMetric(BaseMetric):
    def __init__(self, output_dir: str, ignore_index: Optional[int] = None):
        """
        Args:
            num_classes (int): Number of classes.
            ignore_index (int, optional): Index to ignore. Defaults to None.
        """
        super().__init__()
        self.output_dir = output_dir
        self.ignore_index = ignore_index
        self.reset()

        CLASSES = [
                    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
                    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
                    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
                    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
                    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
                    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
                ]

        CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
        self.IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    def reset(self):
        """Resets the internal evaluation state."""
        # self.preds = []
        # self.targets = []
        # self.dices = []
        self.rles = [] 
        self.filename_and_class = []


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
        
        image_names = [i['img_path'].split('/')[-1] for i in data_samples]

        output_h, output_w = preds.size(-2), preds.size(-1)
        mask_h, mask_w = data_samples[0]['ori_shape']

        if output_h != mask_h or output_w != mask_w:
            preds = F.interpolate(preds, size=(mask_h, mask_w), mode="bilinear")

        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float().detach().cpu()
        # targets = targets.detach().cpu()

        # dice = dice_coef(preds, targets)
        # # self.dices.append(dice)
        # import pdb

        # pdb.set_trace()

        for output,image_name in zip(preds, image_names):
            for c, segm in enumerate(output):
                rle = encode_mask_to_rle(segm)
                self.rles.append(rle)
                self.filename_and_class.append(f"{self.IND2CLASS[c]}_{image_name}")



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


        
        # classes, filename = zip(*[[x.split("_")[0], x.split("_")[-1]] for x in self.filename_and_class])
        classes, filename = zip(*[[x.split("_")[0], '_'.join(x.split('_')[2:])] for x in self.filename_and_class])
        image_name = [os.path.basename(f) for f in filename]

        df = pd.DataFrame({
                "image_name": image_name,
                "class": classes,
                "rle": self.rles,
            })
        

        df.to_csv("output.csv", index=False)

        
        return None

    def evaluate(self, size: int) -> dict:
        """
        Aggregates and computes the final metrics.

        Args:
            size (int): Total number of samples.

        Returns:
            dict: Final evaluation metrics.
        """
        return self.compute_metrics(None)
