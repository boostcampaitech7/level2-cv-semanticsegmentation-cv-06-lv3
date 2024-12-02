# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss


def _expand_onehot_labels_dice(pred: torch.Tensor,
                               target: torch.Tensor) -> torch.Tensor:
    """Expand onehot labels to match the size of prediction.

    Args:
        pred (torch.Tensor): The prediction, has a shape (N, num_class, H, W).
        target (torch.Tensor): The learning label of the prediction,
            has a shape (N, H, W).

    Returns:
        torch.Tensor: The target after one-hot encoding,
            has a shape (N, num_class, H, W).
    """
    num_classes = pred.shape[1]
    one_hot_target = torch.clamp(target, min=0, max=num_classes)
    one_hot_target = torch.nn.functional.one_hot(one_hot_target,
                                                 num_classes + 1)
    one_hot_target = one_hot_target[..., :num_classes].permute(0, 3, 1, 2)
    return one_hot_target


def dice_loss(pred: torch.Tensor,
              target: torch.Tensor,
              weight: Union[torch.Tensor, None],
              eps: float = 1e-3,
              reduction: Union[str, None] = 'mean',
              naive_dice: Union[bool, None] = False,
              avg_factor: Union[int, None] = None,
              ignore_index: Union[int, None] = 255) -> float:
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
            loss defined in the V-Net paper, otherwise, use the
            naive dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        ignore_index (int, optional): The label index to be ignored.
            Defaults to 255.
    """
    if ignore_index is not None:
        num_classes = pred.shape[1]
        pred = pred[:, torch.arange(num_classes) != ignore_index, :, :]
        target = target[:, torch.arange(num_classes) != ignore_index, :, :]
        assert pred.shape[1] != 0  # if the ignored index is the only class
    input = pred.flatten(1)
    target = target.flatten(1).float()
    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def weighted_dice_loss(pred: torch.Tensor,
              target: torch.Tensor,
              weight: Union[torch.Tensor, None],
              class_weight: Union[torch.Tensor, None] = None,  # 추가된 인자
              eps: float = 1e-3,
              reduction: Union[str, None] = 'mean',
              naive_dice: Union[bool, None] = False,
              avg_factor: Union[int, None] = None,
              ignore_index: Union[int, None] = 255) -> float:
    """Calculate dice loss with optional class weights.
    
    Args:
        pred (torch.Tensor): The prediction, has a shape (n, C, ...).
        target (torch.Tensor): The learning label of the prediction,
            has a shape (n, C, ...), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class, 
            has a shape (C,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
            loss defined in the V-Net paper, otherwise, use the
            naive dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        ignore_index (int, optional): The label index to be ignored.
            Defaults to 255.
    """
    if ignore_index is not None:
        num_classes = pred.shape[1]
        pred = pred[:, torch.arange(num_classes) != ignore_index, ...]
        target = target[:, torch.arange(num_classes) != ignore_index, ...]
        assert pred.shape[1] != 0  # if the ignored index is the only class
    
    # Flatten the tensors
    input = pred.flatten(1)
    target = target.flatten(1).float()
    
    # Compute intersection and unions
    intersection = torch.sum(input * target, dim=1)
    if naive_dice:
        denominator = torch.sum(input, dim=1) + torch.sum(target, dim=1) + eps
        dice = (2 * intersection + eps) / denominator
    else:
        denominator = torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + eps
        dice = (2 * intersection) / denominator
    
    loss = 1 - dice
    
    # Apply class weights if provided
    if class_weight is not None:
        if class_weight.dim() == 1 and class_weight.size(0) == pred.size(1):
            loss = loss * class_weight
        else:
            raise ValueError("class_weight should be a 1D tensor with length equal to number of classes")
    
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert len(weight) == pred.size(0)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    
    return loss



@MODELS.register_module()
class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-3,
                 loss_name='loss_dice'):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            ignore_index (int, optional): The label index to be ignored.
                Default: 255.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_dice'.
        """

        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        one_hot_target = target
        if (pred.shape != target.shape):
            one_hot_target = _expand_onehot_labels_dice(pred, target)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            elif pred.shape[1] != 1:
                # softmax does not work when there is only 1 class
                pred = pred.softmax(dim=1)
        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index)

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


@MODELS.register_module()
class WeightedDiceLoss(nn.Module):

    """DiceLoss with optional class weights.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            or softmax. Defaults to True.
        activate (bool): Whether to activate the predictions inside,
            this will disable the inside sigmoid operation.
            Defaults to True.
        reduction (str, optional): The method used
            to reduce the loss. Options are "none",
            "mean" and "sum". Defaults to 'mean'.
        naive_dice (bool, optional): If false, use the dice
            loss defined in the V-Net paper, otherwise, use the
            naive dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power. Defaults to False.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        ignore_index (int, optional): The label index to be ignored.
            Default: 255.
        eps (float): Avoid dividing by zero. Defaults to 1e-3.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_name (str, optional): Name of the loss item. If you want this
            loss item to be included into the backward graph, `loss_` must
            be the prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-3,
                 class_weight=None,  # 추가된 인자
                 loss_name='loss_dice'):
        """Initialize DiceLoss."""
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        self.ignore_index = ignore_index
        self.class_weight = get_class_weight(class_weight)  # 클래스 가중치 처리
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward function."""
        one_hot_target = target
        if pred.shape != target.shape:
            one_hot_target = _expand_onehot_labels_dice(pred, target)
        
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            elif pred.shape[1] != 1:
                # softmax does not work when there is only 1 class
                pred = pred.softmax(dim=1)
        
        # Convert class_weight to tensor if it's a list
        class_weight = self.class_weight
        if isinstance(class_weight, list):
            class_weight = torch.tensor(class_weight, device=pred.device, dtype=pred.dtype)
        
        loss = self.loss_weight * weighted_dice_loss(
            pred,
            one_hot_target,
            weight,
            class_weight=class_weight,  # 클래스 가중치 전달
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index
        )

        return loss

    @property
    def loss_name(self):
        """Loss Name."""
        return self._loss_name