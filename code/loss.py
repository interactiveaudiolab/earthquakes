import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.dist = nn.modules.distance.PairwiseDistance()

    def forward(self, outputs, labels):
        output1 = outputs[:, 0, :]
        output2 = outputs[:, 1, :]
        label = (labels[:, 0, :] * labels[:, 1, :]).sum()
        euclidean_distance = self.dist(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive