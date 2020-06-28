import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets




class MyModel(nn.Module):
    def __init__(self, size):
        super(MyModel, self).__init__()
        self.size = size

        self.conv1 = nn.Conv2d(
                in_channels = 3,
                out_channels = 128,
                kernel_size = 5,
                stride = 1,
                padding = 0
            )

        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(
                in_channels = 128,
                out_channels = 64,
                kernel_size = 5,
                stride = 1,
                padding = 0
            )
        self.bn2 = nn.BatchNorm2d(64)
        
        self.in_dim = int(int(self.size/2-2)/2 - 2)
        #self.in_dim = 143

        self.linear1 = nn.Linear(self.in_dim*self.in_dim*64, 120)
        self.linear2 = nn.Linear(120, 3)

    def forward(self, x):
        # batch * channel(3) * 32 * 32
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(x, 2))
        # batch * channel * 16 * 16

        x = self.bn2(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))
        #print('two', x.shape)
        # batch * channel * 16 * 16
        
        # flatten
        y = x.view(-1, (self.in_dim)*(self.in_dim)*64)
        #print('flatten', y.shape)
        
        y = self.linear1(y)
        y = self.linear2(y)
        #y = nn.sigmoid(y)
        #print('linear', y.shape)

        return y



