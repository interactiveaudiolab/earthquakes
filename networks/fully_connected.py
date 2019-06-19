from torch import nn
import torch.nn.functional as F

class FullyConnected(nn.Module):
    def __init__(self, input_size=100000):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU()
        )

    def forward(self, input_data):
        embedding = self.fc1(input_data)
        embedding = nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding