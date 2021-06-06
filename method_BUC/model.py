# encoding: utf-8

import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class EmbeddingLayer(nn.Module):
    def __init__(self, input_feature_size=2048, embeding_fea_size=1024, dropout=0.5):
        super(self.__class__, self).__init__()

        # embeding
        self.embeding_fea_size = embeding_fea_size
        self.embeding = nn.Linear(input_feature_size, embeding_fea_size)
        self.embeding_bn = nn.BatchNorm1d(embeding_fea_size)
        nn.init.kaiming_normal_(self.embeding.weight, mode='fan_out')
        nn.init.constant_(self.embeding.bias, 0)
        nn.init.constant_(self.embeding_bn.weight, 1)
        nn.init.constant_(self.embeding_bn.bias, 0)
        self.drop = nn.Dropout(dropout)

    def save(self, path):
        torch.save(self.state_dict(), path) 
    
    def load(self, path):
            self.load_state_dict(torch.load(path))

    def forward(self, x):
        if self.training:
            x = self.embeding(x)
            x = self.embeding_bn(x)
            x = F.normalize(x, p=2, dim=1)
            x = self.drop(x) 
            return x
        else:
            return F.normalize(x, p=2, dim=1)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.base = torchvision.models.resnet50(pretrained=True)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)

    def save(self, path):
        torch.save(self.state_dict(), path) 

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x

