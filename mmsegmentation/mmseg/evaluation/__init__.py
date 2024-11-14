# # Copyright (c) OpenMMLab. All rights reserved.
# from .citys_metric import CityscapesMetric
# from .depth_metric import DepthMetric
# from .iou_metric import IoUMetric
# from .custom_metrics import DiceCoefficient

from .metrics import CityscapesMetric, DepthMetric, IoUMetric, DiceCoefficient,LREMetric


__all__ = ['IoUMetric', 'CityscapesMetric', 'DepthMetric','DiceCoefficient', 'LREMetric']