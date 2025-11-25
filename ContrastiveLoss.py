import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self,margin=1.0):
        super().__init__()
        self.margin=margin

    def forward(self, v1, v2,label):

        # label = 0 -> same
        # label = 1 -> different

        distance = F.pairwise_distance(v1, v2)

        pos=(1-label)*0.5*torch.pow(distance, 2)

        neg=label*0.5*torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)

        loss=pos+neg

        return loss.mean()


