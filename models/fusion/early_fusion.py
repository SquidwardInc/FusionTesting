import torch
import torch.nn as nn

class EarlyFusion(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, rgb, ir):

        x = torch.cat([rgb, ir], dim=1)

        return x