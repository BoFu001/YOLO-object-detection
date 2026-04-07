import torch
import torch.nn as nn
import torchvision.models as models


class YOLOv1(nn.Module):
    """
    YOLOv1-style object detection model.
    
    Uses pretrained VGG16 as backbone for feature extraction.
    Backbone weights are frozen, only the head is trained.
    
    Args:
        S (int): grid size (default 7)
        B (int): number of boxes per cell (default 2)
        C (int): number of classes (default 20)
    """

    def __init__(self, S, B, C):
        super().__init__()

        self.S = S
        self.B = B
        self.C = C

        # load pretrained VGG16
        vgg = models.vgg16(weights='IMAGENET1K_V1')

        # use only the convolutional layers as backbone
        self.backbone = vgg.features

        # freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        # YOLO head
        self.head = nn.Sequential(
            nn.MaxPool2d(2),                              # 14x14 → 7x7
            nn.Conv2d(512, 256, kernel_size=1),           # 512 → 256
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, B * 5 + C, kernel_size=1)     # 256 → 30
        )

    def forward(self, x):
        # x: (N, 3, 448, 448)
        x = self.backbone(x)       # (N, 512, 14, 14)
        x = self.head(x)           # (N, 30, 7, 7)
        x = x.permute(0, 2, 3, 1) # (N, 7, 7, 30)
        return x

