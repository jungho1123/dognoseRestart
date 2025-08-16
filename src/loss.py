# loss/contrastive_loss.py
#loss 함수 정의 

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, margin: float = 0.5, reduction: str = "mean"):
        super().__init__()
        assert reduction in ["mean", "sum", "none"]
        self.margin = margin
        self.reduction = reduction

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # cosine similarity → distance
        cos_sim = (z1 * z2).sum(dim=1) 
        dist = 1 - cos_sim  # distance ∈ [0, 2]

        # contrastive loss 계산
        loss = label * dist.pow(2) + (1 - label) * F.relu(self.margin - dist).pow(2)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # [B]

