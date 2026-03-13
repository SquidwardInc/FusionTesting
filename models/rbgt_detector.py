import torch.nn as nn

class RGBTDetector(nn.Module):

    def __init__(self, fusion_module):

        super().__init__()

        self.rgb_backbone = ResNetBackbone()
        self.ir_backbone = ResNetBackbone()

        self.fusion = fusion_module

        self.head = DetectorHead(2048, num_classes=7)

    def forward(self, rgb, ir):

        f_rgb = self.rgb_backbone(rgb)
        f_ir = self.ir_backbone(ir)

        fused = self.fusion(f_rgb, f_ir)

        cls, box = self.head(fused)

        return cls, box