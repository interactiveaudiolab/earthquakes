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

    def forward(self, outputs, labels, weights=None):
        output1 = outputs[:, 0, :]
        output2 = outputs[:, 1, :]
        label = (labels[:, 0, :] * labels[:, 1, :]).sum()
        euclidean_distance = self.dist(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class DeepClusteringLoss(nn.Module):
    def __init__(self):
        """
        Computes the deep clustering loss with weights. Equation (7) in [1].

        References:
            [1] Wang, Z. Q., Le Roux, J., & Hershey, J. R. (2018, April).
            Alternative Objective Functions for Deep Clustering.
            In Proc. IEEE International Conference on Acoustics,  Speech
            and Signal Processing (ICASSP).
        """
        super(DeepClusteringLoss, self).__init__()

    def forward(self, embedding, assignments, weights=None):
        batch_size = embedding.shape[0]
        embedding_size = embedding.shape[-1]
        num_sources = assignments.shape[-1]

        embedding = embedding.view(batch_size, -1, embedding_size)
        assignments = assignments.view(batch_size, -1, num_sources)
        num_points = embedding.shape[1]

        if weights is None:
            weights = embedding.new(batch_size, num_points).fill_(1.0)
        weights = weights.view(batch_size, -1, 1)
        assignments = weights.expand_as(assignments) * assignments
        embedding = weights.expand_as(embedding) * embedding
        norm = ((((weights) ** 2)).sum(dim=1) ** 2).sum()

        vTv = ((embedding.transpose(2, 1) @ embedding) ** 2).sum()
        vTy = ((embedding.transpose(2, 1) @ assignments) ** 2).sum()
        yTy = ((assignments.transpose(2, 1) @ assignments) ** 2).sum()
        loss = (vTv - 2 * vTy + yTy) / norm.detach()
        return loss