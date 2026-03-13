import torch.nn as nn

class DetectorHead(nn.Module):

    def __init__(self, in_channels, num_classes):

        super().__init__()

        self.conv = nn.Conv2d(in_channels, 256, 3, padding=1)

        self.cls = nn.Conv2d(256, num_classes, 1)
        self.box = nn.Conv2d(256, 4, 1)

    def forward(self, x):

        x = self.conv(x)

        cls = self.cls(x)
        box = self.box(x)

        return cls, box