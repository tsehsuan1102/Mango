import torchvision
import torch.nn as nn
from torchvision import models

def get_vgg_model():
    vgg_model = models.vgg16(pretrained=True)
    vgg_model.classifier[6] = nn.Linear(in_features=4096, out_features=3, bias=True)

    return vgg_model





