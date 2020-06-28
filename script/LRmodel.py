import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets




class LRModel(nn.Module):
    def __init__(self, size):
        super(LRModel, self).__init__()
        self.size = size
        self.linear = nn.Linear(size*size*3, 1)

    def forward(self, x):
        y = self.linear(x.view(-1, self.size*self.size*3))
        return y



