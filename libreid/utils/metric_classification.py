# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
import logging

from engine.wheel.metric_simple import Metric


class MetricClassification(Metric):
    def __init__(self, feat_norm='yes'):
        super(MetricClassification, self).__init__()
        self.feat_norm = feat_norm

    def reset(self):
        self.scores = []
        self.targes = []

    def update(self, output):
        scores, targes = output
        self.scores.append(scores)
        self.targes.append(targes)

    def compute(self):
        logger = logging.getLogger(name="reid_baseline")
        scores = torch.cat(self.scores, dim=0)
        targes = torch.cat(self.targes, dim=0)

        acc = (scores.max(1)[1] == targes).float().mean().item()

        return acc
 