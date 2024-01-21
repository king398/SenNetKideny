from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
from torch import nn
import torch

class CombinedLoss(nn.Module):
    def __init__(self, ):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss(mode="multilabel")
        self.bce = SoftBCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, ) -> torch.Tensor:
        dice_loss = self.dice(inputs, targets)
        bce_loss = self.bce(inputs, targets)
        return dice_loss * 0.5 + bce_loss * 0.5
