from torch import nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32, 10))

    def forward_once(self, input_data):
        num_batch = input_data.size(0)
        input_data = input_data.unsqueeze(1)
        output = self.cnn1(input_data)
        output = F.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)
        embedding = self.fc1(output)
        # embedding = nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding

    def forward(self, input_one, input_two):
        output_one = self.forward_once(input_one)
        output_two = self.forward_once(input_two)
        return output_one, output_two