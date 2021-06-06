# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class BalancedSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - num_cameras (int): number of cameras per identity in a batch
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, num_cameras):
        self.__data_source = data_source
        self.__batch_size = batch_size
        self.__num_instances = num_instances
        self.__num_cameras_per_person = num_cameras
        self.__num_pids_per_batch = self.__batch_size // self.__num_instances
        self.__num_instances_per_camera = self.__num_instances // self.__num_cameras_per_person

        #build index dictionary
        self.__index_dict = self.__build_pid_camid_dict()
        self.__pids = list(self.__index_dict.keys())

        # 生成样本列表
        self.__final_idx_list = self.__get_final_idx_list()

    def __len__(self):
        return len(self.__final_idx_list)

    def __iter__(self):
        self.__final_idx_list = self.__get_final_idx_list()
        return iter(self.__final_idx_list)

    def __build_pid_camid_dict(self):
        index_dict = defaultdict(dict)
        for index, (_, pid, camid) in enumerate(self.__data_source):
            index_dict[pid].setdefault(camid, list())
            index_dict[pid][camid].append(index)
        return index_dict

    def __get_final_idx_list(self):
        ## 列举可用pid, 要求至少有指定数量的camera
        index_dict_cpy = copy.deepcopy(self.__index_dict)
        avai_pids = [pid for pid in index_dict_cpy if len(index_dict_cpy[pid])>=self.__num_cameras_per_person]
        final_idxs = []

        ## 
        while len(avai_pids) >= self.__num_pids_per_batch:

            selected_pids = random.sample(avai_pids, self.__num_pids_per_batch)
            for pid in selected_pids:
                camid2index = index_dict_cpy[pid]
                camids = list(camid2index.keys())
                selected_camids = random.sample(camids, self.__num_cameras_per_person)

                for camid in selected_camids:
                    if len(index_dict_cpy[pid][camid]) >= self.__num_instances_per_camera:
                            final_idxs.extend(index_dict_cpy[pid][camid][-self.__num_instances_per_camera:])
                    else:
                        final_idxs.extend(np.random.choice(index_dict_cpy[pid][camid], self.__num_instances_per_camera))

                    ## 如果此camid下的图片小于等于一次采集需要的图片, 则直接删除该camid, 否则只是删除已经采集过的图片
                    if len(index_dict_cpy[pid][camid]) <= self.__num_instances_per_camera:
                        index_dict_cpy[pid].pop(camid)
                    else:
                        index_dict_cpy[pid][camid] = index_dict_cpy[pid][camid][:-self.__num_cameras_per_person]

                if len(index_dict_cpy[pid]) < self.__num_cameras_per_person:
                    index_dict_cpy.pop(pid)
                    avai_pids.remove(pid)

        return final_idxs



## 测试采样器
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    identity_classes = 100
    camera_classes = 6
    batch_size = 64
    num_instances = 4
    num_cameras = 2

    data_source = []
    for i in range(1000):
        pid = random.randint(0, identity_classes-1)
        camid = random.randint(0, camera_classes-1)
        data_source.append((0, pid, camid))

    sampler = BalancedSampler(data_source, batch_size, num_instances, num_cameras)



    train_loader = DataLoader(
    data_source, batch_size=batch_size,
    sampler=sampler,
    )

    print(len(train_loader))

    res = []
    for epoch in range(3):
        for i,batch in enumerate(train_loader):
            if i == 0:
                res.append(batch)

    


