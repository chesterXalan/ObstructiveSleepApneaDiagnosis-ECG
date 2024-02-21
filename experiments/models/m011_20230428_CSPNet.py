"""
changelogs:
    stem:
        change C1 and C2 from 48 and 16 to 48 and 32
"""

import torch
from torch import nn


def calc_padding(kernel_size, padding):
    if padding == 'same':
        return kernel_size//2
    elif padding == 'valid':
        return 0

def conv1d(Ci, Co, kernel_size, stride, padding):
    module = nn.Conv1d(Ci, Co,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=calc_padding(kernel_size, padding),
                       bias=False)
    return module

def conv1d_bn(Ci, Co, kernel_size, stride, padding='same'):
    module = nn.Sequential(
        conv1d(Ci, Co, kernel_size, stride, padding),
        nn.BatchNorm1d(Co)
    )
    return module

def conv1d_bn_relu(Ci, Co, kernel_size, stride, padding='same'):
    module = conv1d_bn(Ci, Co, kernel_size, stride, padding)
    module.append(nn.ReLU(inplace=True))
    return module

class ResNetBlock1(nn.Module):
    '''
    residual path: 
        Conv * 3:
        kernel =  3,  3, 1
        filter = *2, *2, 1 (對應於 Ci)
    '''
    def __init__(self, Ci):
        super().__init__()
        C1 = C2 = Ci*2
        self.layers = nn.Sequential(
            conv1d_bn_relu(Ci, C1, 3, 1),
            conv1d_bn_relu(C1, C2, 3, 1),
            conv1d_bn(C2, Ci, 1, 1)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        y = self.layers(x)
        y = torch.add(x, y)
        y = self.relu(y)  
        return y

class ResNetBlock2(nn.Module):
    '''
    residual path: 
        Conv * 3:
        kernel =  1,  3, 1
        filter = /2, /2, 1 (對應於 Ci)
    '''
    def __init__(self, Ci):
        super().__init__()
        C1 = C2 = Ci//2
        self.layers = nn.Sequential(
            conv1d_bn_relu(Ci, C1, 1, 1),
            conv1d_bn_relu(C1, C2, 3, 1),
            conv1d_bn(C2, Ci, 1, 1)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        y = self.layers(x)
        y = torch.add(x, y)
        y = self.relu(y)  
        return y

class CSPNetBlock(nn.Module):
    def __init__(self, Ci, Co, Res_block, kernel_size, stride):
        super().__init__()
        Cn = Ci//2
        self.layers1 = conv1d_bn_relu(Ci, Cn, 1, 1)
        self.layers2 = nn.Sequential(
            conv1d_bn_relu(Ci, Cn, 1, 1),
            Res_block(Cn),
            Res_block(Cn),
            conv1d_bn_relu(Cn, Cn, 1, 1)
        )
        self.layers3 = conv1d_bn_relu(Ci, Co, kernel_size, stride)
        
    def forward(self, x):
        y1 = self.layers1(x)
        y2 = self.layers2(x)
        y = torch.cat((y1, y2), 1)
        y = self.layers3(y)
        return y

class MultiPath(nn.Module):
    def __init__(self, Ci, Co):
        super().__init__()
        self.layers1 = nn.Sequential(
            conv1d_bn_relu(Ci, 48, 19, 10, 'valid'),
            conv1d_bn_relu(48, 48, 19, 10, 'valid')
        )
        self.layers2 = nn.Sequential(
            conv1d_bn_relu(Ci, 32, 9, 5, 'valid'),
            conv1d_bn_relu(32, 32, 9, 5, 'valid'),
            conv1d_bn_relu(32, 32, 7, 4, 'valid')
        )
        self.layers3 = conv1d_bn_relu(80, Co, 1, 1)

    def forward(self, x):
        y1 = self.layers1(x)
        y2 = self.layers2(x)
        y = torch.cat((y1, y2), 1)
        y = self.layers3(y)
        return y
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = MultiPath(1, 64)
        self.body = nn.Sequential(
            CSPNetBlock(64, 64, ResNetBlock1, 15, 8),
            CSPNetBlock(64, 128, ResNetBlock1, 9, 5),
            CSPNetBlock(128, 128, ResNetBlock2, 1, 1)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1), # globle average pooling
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)
        )
        
    def forward(self, x):
        y = self.stem(x)
        y = self.body(y)
        y = self.head(y)
        return y
