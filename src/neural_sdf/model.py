import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

class SDFNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(3, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 509), nn.ReLU(),
        )

        self.second_layer = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Tanh(),
        )
    
    def forward(self, x):
        h = self.first_layer(x)
        return self.second_layer(torch.cat([h, x], dim=1))