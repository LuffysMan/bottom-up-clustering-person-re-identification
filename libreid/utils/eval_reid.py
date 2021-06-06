# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from .iotools import get_time_cost


@get_time_cost
def eval_func_parallelly_gpu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """ A parallel version of funcition 'eval_func'; Implemented by pytorch;  Running on GPU; It is really gpu memory consuming when distmat getting large.
    :param:
        :distmat: Tensor with size of (m,n) which indicates the similarity of query and gallery; m: number of query samples, n: number of gallery samples;
        :q_pids: Tensor with size of (m,) in which is pids of query samples.
        :g_pids: Tensor with size of (n,) in which is pids of gallery samples.
        :q_camids: Tensor with size of (m,) in which is camids of query samples.
        :g_camids: Tensor with size of (b,) in which is camids of gallery samples.
        :max_rank: The up-bound amount of cmc elements.
    :return: 
        :cmc: 
        :mAP: mean average precision.
    """
    device_id = distmat.device

    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    # set distmat[i,j] = max_dist+1.0 that i-th query and j-th gallery have the same pid and camid, so that it won't affect the final result when eliminate those sample later.
    mask = (q_pids[:,None] == g_pids[None,:]) & (q_camids[:,None]==g_camids[None,:])
    distmat[mask] = distmat.max() + 1.0
    mask = mask.int().sort(axis=1)[0]
    mask = mask.bool()

    # get the match matrix
    indices = torch.argsort(distmat, axis=1)                                   # (m,n)
    matches = (g_pids[indices] == q_pids[:, None]).int()                        # (m,n)

    # eliminate the influence of those samples in gallery which have the same pid and camid with query,  by set correspondent val to 0.
    matches[mask] = 0

    # eliminate invalid query
    matches_valid = matches.sum(axis=1).bool()
    matches = matches[matches_valid]
    num_valid_query = matches.shape[0]

    assert num_valid_query > 0, "Error: all query identities do not appear in gallery"

    # eliminate gallery samples that have same pid and same camid with query

    # compute cmc curve for each query
    cmc = matches.cumsum(axis=1)                                                # (m,n)
    denominator_cmc = torch.ones((num_valid_query, num_g), dtype=torch.int32, device=device_id).cumsum(axis=1)    # (m,n)
    AP = cmc/denominator_cmc*matches                                            # (m,n)
    AP = AP.sum(axis=1)/matches.sum(axis=1)                                     # (m,)
    mAP = AP.sum()/num_valid_query

    # compute cmc
    cmc[cmc>1] = 1
    cmc = cmc[:, :max_rank]

    # num_valid_query = cmc.shape[0]
    mean_cmc = cmc.sum(axis=0)/num_valid_query

    return mean_cmc, mAP


@get_time_cost
def eval_func_parallel_cpu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """ A parallel version of funcition 'eval_func'; Implemented by numpy;  Running on cpu;
    :param:
        :distmat: Tensor with size of (m,n) which indicates the similarity of query and gallery; m: number of query samples, n: number of gallery samples;
        :q_pids: Tensor with size of (m,) in which is pids of query samples.
        :g_pids: Tensor with size of (n,) in which is pids of gallery samples.
        :q_camids: Tensor with size of (m,) in which is camids of query samples.
        :g_camids: Tensor with size of (b,) in which is camids of gallery samples.
        :max_rank: The up-bound amount of cmc elements.
    :return: 
        :cmc: 
        :mAP: mean average precision.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    # set distmat[i,j] = max_dist+1.0 that i-th query and j-th gallery have the same pid and camid, so that it won't affect the final result when eliminate those sample later.
    mask = (q_pids[:,None] == g_pids[np.newaxis,:]) & (q_camids[:,None]==g_camids[None,:])
    distmat[mask] = distmat.max() + 1.0
    mask.sort(axis=1)
    
    # get the match matrix
    indices = np.argsort(distmat, axis=1)                                   # (m,n)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)   # (m,n)

    # eliminate the influence of those samples in gallery which have the same pid and camid with query,  by set correspondent val to 0.
    matches[mask] = 0

    # eliminate invalid query
    matches_valid = matches.sum(axis=1).astype(np.bool)
    matches = matches[matches_valid]
    num_valid_query = matches.shape[0]

    assert num_valid_query > 0, "Error: all query identities do not appear in gallery"

    # eliminate gallery samples that have same pid and same camid with query

    # compute cmc curve for each query
    cmc = matches.cumsum(axis=1)                                                # (m,n)
    denominator_cmc = np.ones((num_valid_query, num_g), dtype=np.int32).cumsum(axis=1)    # (m,n)
    AP = cmc/denominator_cmc*matches                                            # (m,n)
    AP = AP.sum(axis=1)/matches.sum(axis=1)                                     # (m,)
    mAP = AP.sum()/num_valid_query

    # compute cmc
    cmc[cmc>1] = 1
    cmc = cmc[:, :max_rank]

    # num_valid_query = cmc.shape[0]
    cmc = cmc.sum(axis=0)/num_valid_query

    return cmc, mAP


@get_time_cost
def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """evaluation with market1501 metric. Note that for each query identity, its gallery images from the same camera view are discarded.
    :param:
        :distmat: Tensor with size of (m,n) which indicates the similarity of query and gallery; m: number of query samples, n: number of gallery samples;
        :q_pids: array with size of (m,) in which is pids of query samples.
        :g_pids: array with size of (n,) in which is pids of gallery samples.
        :q_camids: array with size of (m,) in which is camids of query samples.
        :g_camids: array with size of (b,) in which is camids of gallery samples.
        :max_rank: The up-bound amount of cmc elements.
    :return: 
        :cmc: 
        :mAP: mean average precision.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
