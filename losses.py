#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from torch import Tensor, einsum

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss

class Weighted_CrossEntropy():
    def _init_(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk'] 
        self.class_weights = kwargs['weight']
        
        print(f"Initialized {self._class.name_} with {kwargs}")

    def _call_(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        weighted_log_p = log_p * self.class_weights.view(1, -1, 1, 1)

        loss = - einsum("bkwh,bkwh->", mask, weighted_log_p)
        loss /= mask.sum() + 1e-10

        return loss

class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)



class GeneralizedDice():
    def __init__(self, **kwargs):
        self.idk: List[int] = kwargs["idk"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idk, ...].type(torch.float32)
        tc = target[:, self.idk, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
        union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

        divided: Tensor = 1 - 2 * (einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)

        loss = divided.mean()

        return loss


class MulticlassDice():
    def __init__(self, **kwargs):
        self.idk: List[int] = kwargs["idk"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        # Predicted and true class probabilities for the given classes (idk)
        pc = probs[:, self.idk, ...].type(torch.float32)  # Predicted probs
        tc = target[:, self.idk, ...].type(torch.float32)  # True labels

        # Shape: batch_size, num_classes, height, width
        batch_size, num_classes, h, w = pc.shape

        # Flatten height and width into a single dimension (number of pixels)
        pc = pc.view(batch_size, num_classes, -1)
        tc = tc.view(batch_size, num_classes, -1)

        # Intersection (numerator): sum of element-wise multiplication between target and prediction
        intersection: Tensor = torch.sum(pc * tc, dim=-1)

        # Union (denominator): sum of predictions and target values for each class
        union: Tensor = torch.sum(pc, dim=-1) + torch.sum(tc, dim=-1)

        # Dice coefficient for each class (for each batch)
        dice_per_class: Tensor = 2 * intersection / (union + 1e-10)

        # Averaging over classes and batch
        dice_loss = 1 - dice_per_class.mean()
        
        print("Dice loss:", dice_loss)

        return dice_loss


class DiceLoss():
    def __init__(self, **kwargs):
        self.idk: List[int] = kwargs["idk"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idk, ...].type(torch.float32)
        tc = target[:, self.idk, ...].type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

        divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss

class TverskyLoss:
    def __init__(self, alpha=0.5, beta=0.5, **kwargs):
        """
        Initializes the TverskyLoss.

        Parameters:
        - alpha (float): Weight of false positives.
        - beta (float): Weight of false negatives.
        - kwargs: Additional keyword arguments (e.g., 'idk' for class filtering).
        """
        self.alpha = alpha
        self.beta = beta
        self.idk: List[int] = kwargs.get("idk", [])
        print(f"Initialized {self.__class__.__name__} with alpha={alpha}, beta={beta}, idk={self.idk}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        """
        Computes the Tversky loss.

        Parameters:
        - probs (Tensor): Predicted probabilities (after softmax or sigmoid).
        - target (Tensor): Ground truth labels.

        Returns:
        - loss (Tensor): Computed Tversky loss.
        """
        assert probs.shape == target.shape, "Shape mismatch between predictions and targets."
        assert simplex(probs), "Predictions are not simplex."
        assert sset(target, [0, 1]), "Targets must be binary."

        if self.idk:
            probs = probs[:, self.idk, ...]
            target = target[:, self.idk, ...]

        # Flatten the tensors to (batch_size, num_classes, -1) for easier computation
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)

        # True Positives, False Positives & False Negatives
        TP = (probs_flat * target_flat).sum(dim=2)
        FP = (probs_flat * (1 - target_flat)).sum(dim=2)
        FN = ((1 - probs_flat) * target_flat).sum(dim=2)

        # Compute Tversky index
        Tversky_index = TP / (TP + self.alpha * FP + self.beta * FN + 1e-10)

        # Tversky loss is 1 - Tversky index
        loss = 1 - Tversky_index

        return loss.mean()