import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

NUM_FREQS = 6

class SDFNet(nn.Module):
    def __init__(self):
        super().__init__()

        # reserve for Fourier positional encoding
        self.register_buffer("freqs", (2 ** torch.arange(NUM_FREQS)) * torch.pi)
        enc_dim = 3 + 6 * NUM_FREQS

        self.first_layer = nn.Sequential(
            nn.Linear(enc_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512 - enc_dim), nn.ReLU(),
        )

        self.second_layer = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def forward(self, x):
        scaled = x[..., None] * self.freqs
        waves = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        waves = waves.flatten(start_dim=1)
        e = torch.cat([x, waves], dim=-1)

        h = self.first_layer(e)
        return self.second_layer(torch.cat([h, e], dim=1))