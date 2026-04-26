import torch.nn as nn
import torchvision.models as models


class YOLOv1Dropout(nn.Module):
    """
    YOLOv1 with Dropout in head.
    Used in Experiment 3.
    """

    def __init__(self, S, B, C):
        super().__init__()

        self.S = S
        self.B = B
        self.C = C

        vgg = models.vgg16(weights='IMAGENET1K_V1')
        self.backbone = vgg.features

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),                      # dropout
            nn.Conv2d(256, B * 5 + C, kernel_size=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)
        return x