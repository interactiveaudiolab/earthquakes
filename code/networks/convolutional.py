from torch import nn
import torch.nn.functional as F
import torch

class DilatedConvolutional(nn.Module):
    """[summary]
        
        Args:
            embedding_size ([type]): [description]
    """
    def __init__(self, embedding_size):
        super(DilatedConvolutional, self).__init__()
        dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        channels = [1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
        self.main = nn.Sequential()
        for i, d in enumerate(dilations):
            self.main.add_module(str(i) + '_batch_norm', nn.BatchNorm1d(channels[i]))
            self.main.add_module(str(i), nn.Conv1d(in_channels=channels[i],
                                                   out_channels=channels[i + 1],
                                                   kernel_size=2,
                                                   dilation=d,
                                                   padding=d))
            self.main.add_module(str(i) + '_relu', nn.ReLU())
        self.main.add_module('pool', nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(channels[-1], embedding_size)

    def forward(self, input):
        output = self.main(input).squeeze(-1)
        output = self.fc(output)
        output = nn.functional.normalize(output, dim=-1,p=2)
        return output