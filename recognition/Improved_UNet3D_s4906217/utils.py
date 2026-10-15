"""Loss and metrics for HipMRI segmentation (PyTorch only).

    DiceCELoss         cross-entropy + soft Dice, the training loss
    per_class_scores   Dice and IoU per class of a hard prediction, the evaluation metrics

Both work on 2D (B, C, X, Y) and 3D (B, C, X, Y, Z) logits alike.
"""
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class DiceCELoss(nn.Module):
    """Cross-entropy + soft Dice for a heavily imbalanced multi-class segmentation.

    The prostate is 0.10% of the voxels. Cross-entropy alone is dominated by the 60% background
    and the 35% body, and a network that predicts 'no prostate' everywhere already gets a small
    loss. The soft Dice term scores every class by its overlap, whatever its size, so the
    prostate counts as much as the body. Cross-entropy keeps the gradients smooth and well
    behaved early in training, where the Dice gradient is noisy.

    * The Dice is computed per class over the whole batch (not per sample), which keeps it
      defined for a patch that happens to contain no prostate.
    * With include_background = False (the default) the background class does not enter the
      Dice average; it is still part of the cross-entropy.
    * `smooth` keeps a class that is absent from both the prediction and the target at a Dice of
      about 1 instead of 0 / 0.
    """

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1e-5,
                 include_background: bool = False):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits (B, C, ...) of any float type, target (B, ...) of class indices."""
        loss = logits.new_zeros(()).float()
        if self.ce_weight:
            loss = loss + self.ce_weight * F.cross_entropy(logits.float(), target)
        if self.dice_weight:
            probs = logits.float().softmax(dim=1)
            onehot = F.one_hot(target, logits.shape[1]).movedim(-1, 1).to(probs.dtype)
            dims = [0] + list(range(2, logits.dim()))   # everything except the class axis
            inter = (probs * onehot).sum(dims)
            denom = probs.sum(dims) + onehot.sum(dims)
            dice = (2 * inter + self.smooth) / (denom + self.smooth)   # (C,)
            if not self.include_background:
                dice = dice[1:]
            loss = loss + self.dice_weight * (1.0 - dice.mean())
        return loss


def per_class_scores(pred: torch.Tensor, target: torch.Tensor,
                     num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dice and IoU of every class between a hard prediction and the ground truth.

    pred and target are integer class maps of the same shape (a whole volume, for the volume
    metrics of the README). Returns two float tensors of length num_classes. A class that is
    absent from both maps has no overlap to measure and scores NaN, so that averaging over
    volumes with torch.nanmean skips it instead of counting it as a perfect or a zero score.

    Dice = 2 TP / (2 TP + FP + FN),  IoU = TP / (TP + FP + FN).
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ in shape")
    # one pass: confusion[t, p] = number of voxels with ground truth t predicted as p
    pairs = target.reshape(-1).long() * num_classes + pred.reshape(-1).long()
    confusion = torch.bincount(pairs, minlength=num_classes ** 2).reshape(num_classes, num_classes).double()
    tp = confusion.diagonal()
    gt = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    nan = torch.full_like(tp, float("nan"))
    dice = torch.where(gt + predicted > 0, 2 * tp / (gt + predicted), nan)
    iou = torch.where(gt + predicted > 0, tp / (gt + predicted - tp), nan)
    return dice.float(), iou.float()
