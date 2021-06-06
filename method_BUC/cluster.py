from math import pi
import pickle
import torch
import heapq
import logging
import copy
import numpy as np

from libreid.utils.math_tools import euclidean_dist
from libreid.utils.iotools import get_time_cost


class UnionFind(object):
    r""" Use disjoint set to store labels of all samples. Which is really convenient to merge clusters and find labels.
    
    Args:
        labels(tensor): 1D tensor that contains the label of each sample.

    * attr: __parent(tensor): Store the label of each sample.
    * attr: __size(list): Store the amount of nodes .
    * attr: __changed(boolean): If true, the __parent has been changed, you need to update __parent before you fetch a batch of label
    * attr: __parent_contains(list):  
    """
    def __init__(self, num_samples, device="cuda:0"):
        self.__parent = torch.as_tensor(range(num_samples), dtype=torch.int64, device=device)
        self.__parent_to_label = torch.as_tensor(range(num_samples), dtype=torch.int64, device=device)
        self.__size = num_samples
        self.__num_clusters = num_samples
        self.__changed = False

    def find(self, index):
        if index != self.__parent[index]:
            self.__parent[index] = self.find(self.__parent[index])

        return self.__parent[index]

    def merge(self, index0, index1):
        parent0 = self.find(index0)
        parent1 = self.find(index1)
        if parent0 != parent1:
            self.__parent[parent1] = parent0
            self.__num_clusters -= 1
            self.__changed = True
            return True
        else:
            return False
    
    def __update(self):
        r""" Update __parent that each sample is linked directly to its label.
        """
        if self.__changed:
            for idx in range(self.__parent.size(0)):
                self.find(idx)
            self.__changed = False

    def update_lable_map(self):
        r""" The labels in __parent are not continuous numbers start from 0. 
        So we need to relabel them and store in __parent_to_label.
        """
        self.__update()
        all_labels = self.__parent_to_label[self.__parent].cpu().numpy()
        labels_left = list(set(all_labels))
        labels_left.sort()
        relabel_map = dict(zip(labels_left, range(len(labels_left))))
        for i in range(self.__size):
            self.__parent_to_label[i] = relabel_map[all_labels[i]]

    def get_labels(self, sample_ids):
        self.__update()
        return self.__parent_to_label[self.__parent[sample_ids]]

    def num_clusters(self):
        return self.__num_clusters

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)
            
    def for_debug_count_clusters(self):
        parent_set = set()
        for index in range(self.__parent.size(0)):
            parent = self.for_debug_find(index)
            parent_set.add(parent.item())
        return len(parent_set)

    
    def for_debug_find(self, index):
        route = [index]
        parent = self.__parent[index]
        route.append(parent)
        while (parent != self.__parent[parent]):
            self.__parent[index] = self.__parent[parent]
            parent = self.__parent[index]
            route.append(parent)

        print("-->".join(map(str, route)))
        return self.__parent[index]

    def get_purity(self, true_labels, num_classes):
        bucket = [[0] * num_classes for _ in range(self.num_clusters())]
        for idx in range(self.__size):
            pseudo_label = self.__parent_to_label[self.__parent[idx]]
            true_label = true_labels[idx]
            bucket[pseudo_label][true_label] += 1
        purity = sum([max(bucket[k]) for k in range(self.num_clusters())]) / self.__size
        return purity



class UnionFindWithSnapshot(UnionFind):
    
    def __init__(self, num_samples, device="cuda:0"):
        super().__init__(num_samples, device=device)
        self.__parent_contains = [[i] for i in range(num_samples)]
        self.__snapshots = []

    def merge(self, index0, index1):
        parent0 = self.find(index0)
        parent1 = self.find(index1)
        if super().merge(index0, index1):
            self.__parent_contains[parent0].extend(self.__parent_contains[parent1])

    def save_snapshots(self):
        self.__snapshots.append(copy.deepcopy(self.__parent_contains))

    @property
    def depth(self):
        return len(self.__snapshots)


def get_max_joint(pids, num_classes):
    bucket = [[0] for _ in range(num_classes)]
    for pid in pids:
        bucket[pid] += 1
    return max(bucket)
    

def get_purity_by_one_cluster(cluster, num_classes):
    return get_max_joint(cluster) / len(cluster)

    
def get_purity_by_clusters(clusters, num_classes):
    num_samples = 0
    num_max_joints = 0
    for cluster in clusters:
        num_samples += len(cluster)
        num_max_joints += get_max_joint(cluster, num_classes)
    num_max_joints = num_max_joints / num_samples
    return num_max_joints
class FixedCapacityBigTopHeap(object):
    def __init__(self, capacity):
        self.__data = []
        self.__size = 0
        self.__capacity = capacity
    
    def push(self, item, key_getter=None):
        key_getter = key_getter if key_getter else lambda x: -x
        key = key_getter(item)
        if self.__size == self.__capacity:
            if key > key_getter(self.top()):
                heapq.heappop(self.__data)
                heapq.heappush(self.__data, (key, item))
        else:
            heapq.heappush(self.__data, (key, item))
            self.__size += 1

    def pop(self):
        if self.__size == 0:
            raise IndexError
        self.__size -= 1
        return heapq.heappop(self.__data)[1]

    def top(self):
        if self.__size == 0:
            raise IndexError
        return self.__data[0][1]

    def empty(self):
        return self.__size == 0

    def size(self):
        return self.__size

    def capacity(self):
        return self.__capacity


@get_time_cost
def get_distance_matrix(model, databox_0):
    logger = logging.getLogger("reid_baseline")
    logger.info("getting distance matrix")

    all_features = []
    all_sample_ids = []
    model.cuda()
    model.eval()
    with torch.no_grad():
        for batch in databox_0.train_loader:
            imgs, *_, sample_ids = batch
            imgs = imgs.to("cuda:0")
            all_features.append(model(imgs))
            all_sample_ids.extend(sample_ids)
    all_features = torch.cat(all_features, dim=0)
    distmat = euclidean_dist(all_features, all_features)
    return all_sample_ids, distmat


@get_time_cost
def get_pairs_ordered_by_pair_dist(Y, all_sample_ids, distmat):
    r""" get index pairs of samples, ordered by distance of each pair
    """
    logger = logging.getLogger("reid_baseline")
    logger.info("getting pair list ordered by pair distance")

    max_distance = distmat.max()
    distmat.add_(torch.tril(max_distance * torch.ones_like(distmat)))
    all_sample_labels = Y.get_labels(all_sample_ids)
    mask = all_sample_labels[:, None] == all_sample_labels[None, :]
    distmat.add_(mask.float() * 100000)
    distmat = distmat.view(-1)
    indices = distmat.argsort()
    num_samples = len(all_sample_ids)
    idx0, idx1 = np.unravel_index(
        indices.cpu().numpy(), (num_samples, num_samples))
    return idx0, idx1


def clustering(databox_0, NUM_TO_MERGE, disjoint_set, model):
    # clustering, 计算所有簇之间的距离(使用文中提出的minimum distance criterion), 选取距离最小的num_to_merge对簇进行合并
        # 求样本距离矩阵 NxN
    logger = logging.getLogger("reid_baseline")
    logger.info("clustering")

    all_sample_ids, distmat = get_distance_matrix(model, databox_0)
    indices0, indices1 = get_pairs_ordered_by_pair_dist(disjoint_set, all_sample_ids, distmat)
    del distmat

    # merge clusters with the minimum distancelogger.info("merging clusters")
    num_clusters_before = disjoint_set.num_clusters()
    idx = 0
    while num_clusters_before - disjoint_set.num_clusters() < NUM_TO_MERGE:
        disjoint_set.merge(all_sample_ids[indices0[idx]],
                    all_sample_ids[indices1[idx]])
        idx += 1
    num_clusters_before -= NUM_TO_MERGE
    logger.info("{} clusters left after merging".format(num_clusters_before))


@get_time_cost
def get_distance_matrix_for_test(model, databox_0):
    logger = logging.getLogger("reid_baseline")
    logger.info("delete me after testting!")
    logger.info("getting distance matrix")

    all_features = []
    all_sample_ids = []
    all_sample_pids = []
    model.cuda()
    model.eval()
    with torch.no_grad():
        for batch in databox_0.train_loader:
            imgs, pids, *_, sample_ids = batch
            imgs = imgs.to("cuda:0")
            pids = pids.to("cuda:0")
            all_features.append(model(imgs))
            all_sample_pids.append(pids)
            all_sample_ids.extend(sample_ids)
    all_features = torch.cat(all_features, dim=0)
    all_sample_pids = torch.cat(all_sample_pids, dim=0)
    distmat = euclidean_dist(all_features, all_features)
    return all_sample_ids, all_sample_pids, distmat


@get_time_cost
def get_pairs_ordered_by_pair_dist_for_test(Y, all_sample_ids, all_sample_pids, distmat):
    r""" get index pairs of samples, ordered by distance of each pair
    """
    logger = logging.getLogger("reid_baseline")
    logger.info("delete me after testting!")
    logger.info("getting pair list ordered by pair distance")

    max_distance = distmat.max()
    distmat.add_(torch.tril(max_distance * torch.ones_like(distmat)))
    all_sample_labels = Y.get_labels(all_sample_ids)
    mask = all_sample_labels[:, None] == all_sample_labels[None, :]
    mask |= ~(all_sample_pids[:, None] == all_sample_pids[None, :])
    distmat.add_(mask.float() * 100000)
    distmat = distmat.view(-1)
    indices = distmat.argsort()
    num_samples = len(all_sample_ids)
    idx0, idx1 = np.unravel_index(
        indices.cpu().numpy(), (num_samples, num_samples))
    return idx0, idx1


def clustering_for_test(databox_0, NUM_TO_MERGE, Y, model):
    # clustering, 计算所有簇之间的距离(使用文中提出的minimum distance criterion), 选取距离最小的num_to_merge对簇进行合并
        # 求样本距离矩阵 NxN
    logger = logging.getLogger("reid_baseline")
    logger.info("delete me after testting!")
    logger.info("clustering")

    all_sample_ids, all_sample_pids, distmat = get_distance_matrix_for_test(model, databox_0)
    indices0, indices1 = get_pairs_ordered_by_pair_dist_for_test(Y, all_sample_ids, all_sample_pids, distmat)
    del distmat, all_sample_pids

    # merge clusters with the minimum distancelogger.info("merging clusters")
    num_clusters_before = Y.num_clusters()
    idx = 0
    while num_clusters_before - Y.num_clusters() < NUM_TO_MERGE:
        Y.merge(all_sample_ids[indices0[idx]],
                    all_sample_ids[indices1[idx]])
        idx += 1
    num_clusters_before -= NUM_TO_MERGE
    logger.info("{} clusters left after merging".format(num_clusters_before))
    return Y