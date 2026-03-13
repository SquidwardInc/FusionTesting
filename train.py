from torch.utils.data import dataloader

from models.rbgt_detector import RGBTDetector
from models.fusion.early_fusion import EarlyFusion
import torch
from torch import dataloader
model = RGBTDetector(
    fusion_module=EarlyFusion(2048)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)
epochs = 100





for epoch in range(epochs):

    for rgb, ir, target in dataloader:

        cls, box = model(rgb, ir)

        loss = compute_loss(cls, box, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()