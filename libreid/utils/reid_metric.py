# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
import logging

from libreid.engine.metric_simple import Metric
from libreid.utils.eval_reid import eval_func, eval_func_parallel_cpu, eval_func_parallelly_gpu
from libreid.utils.re_ranking import re_ranking
from libreid.utils.math_tools import cosine_dist, euclidean_dist


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        return self.__compute_parallel_gpu()

    def __compute_parallel_cpu(self):
        """ computing cmc and mAP running on cpu
        """
        logger = logging.getLogger(name="reid_baseline")

        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            logger.info("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])

        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        # compute distance matrix
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)

        # distmat = cosine_dist(qf, gf)
        distmat = euclidean_dist(qf, gf)

        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func_parallel_cpu(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP

    def __compute_parallel_gpu(self):
        """ computing cmc and mAP running on gpu
        """
        logger = logging.getLogger(name="reid_baseline")

        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            logger.info("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = torch.as_tensor(self.pids[:self.num_query]).cuda()
        q_camids = torch.as_tensor(self.camids[:self.num_query]).cuda()

        # gallery
        gf = feats[self.num_query:]
        g_pids = torch.as_tensor(self.pids[self.num_query:]).cuda()
        g_camids = torch.as_tensor(self.camids[self.num_query:]).cuda()

        # compute distance matrix
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
        
        cmc, mAP = eval_func_parallelly_gpu(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP