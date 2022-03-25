from torchvision import models
import torch.nn as nn


class HandSegModel(nn.Module):
    def __init__(self):
        super(HandSegModel, self).__init__()
        deeplab = models.segmentation.deeplabv3_resnet50(pretrained=0, progress=1, num_classes=2)
        self.dl = deeplab

    def forward(self, x):   # defines the computation performed at every call
        y = self.dl(x)['out']
        return y
