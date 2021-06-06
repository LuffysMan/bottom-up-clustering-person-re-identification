import sys
sys.path.append('.')
import argparse
import os
import os.path as osp
import torch
import math
import logging
from torch.backends import cudnn
from torch.optim.lr_scheduler import MultiStepLR

from libreid.engine.engine_simple import State
from libreid.utils.logger import setup_logger
from libreid.utils.reid_metric import R1_mAP
from method_BUC.trainer import Trainor, Evaluator
from method_BUC.build_data import make_data_loader
from method_BUC.model import FeatureExtractor, EmbeddingLayer
from method_BUC.config import BUC as config
from method_BUC.cluster import UnionFind, clustering
from method_BUC.loss import ExLoss


class Pipeline(object):
    r""" This class describe the whole pipeline of method BUC. 
    """
    def __init__(self, cfg):
        self._logger = logging.getLogger("BUC")
        self._cfg = cfg
        self._events_after_step = []
        self._archive_list = []
        self._pipe_for_train = None
        self._state = State(best_rank1=0.0, best_mAP=0.0, best_step=-1,
                            current_rank1=0.0, current_mAP=0.0, current_step=-1,
                            start_step=-1)

        # Prepare dataset
        self._databox = make_data_loader(cfg)
        self._NUM_TO_MERGE = math.ceil(self._databox.num_samples * cfg.CLUSTER_MERGE_PERCENT)

        # Prepare initial cluster identities for samples. Each sample is considered as a cluster initially.
        self._disjoint_set = UnionFind(self._databox.num_samples, device=cfg.MODEL.DEVICE)

        # Prepare model
        self._feature_extractor = FeatureExtractor()
        self._embedding_layer = EmbeddingLayer()
        self._repell_loss = ExLoss(t=10)
        self._repell_loss.create_lookup_table(self.num_clusters)

        self._add_archive_modules(self._disjoint_set, self._feature_extractor, self._embedding_layer, self._repell_loss)

        if cfg.RESUME:
            self._resume()

        if cfg.TRAIN:
            self.add_handler_after_step(self._save_checkpoint, [self._feature_extractor, self._embedding_layer, self._disjoint_set, self._repell_loss])
            self.add_handler_after_step(self.evaluate)  # Cannot change the order: self.evaluate -> self._save_best_modules
            self.add_handler_after_step(self._save_best_modules, [self._feature_extractor])
            self.add_handler_after_step(self.get_purity)
    @property
    def num_clusters(self):
        return self._disjoint_set.num_clusters()

    @property
    def pipe_for_train(self):
        if self._pipe_for_train is None:
            self._pipe_for_train = torch.nn.Sequential(self._feature_extractor, self._embedding_layer)
        return self._pipe_for_train

    def _add_archive_modules(self, *args):
        self._archive_list.extend(list(args))

    def _get_archive_modules(self):
        return self._archive_list

    def _save_checkpoint(self, module_list) -> None:
        for module in module_list:
            path_to_module = osp.join(self._cfg.OUTPUT_DIR,
                "BUC_{}_{}".format(self._state.current_step, module.__class__.__name__))
            module.save(path_to_module)

    def _save_best_modules(self, module_list) -> None:
        if self._state.current_rank1 > self._state.best_rank1:
            for module in module_list:
                path_to_last_best_module = osp.join(
                    self._cfg.OUTPUT_DIR,
                    "BUC_best_{:.3f}_{}_{}".format(self._state.best_rank1, self._state.best_step, module.__class__.__name__))
                if osp.exists(path_to_last_best_module):
                    os.remove(path_to_last_best_module)
            for module in module_list:
                path_to_best_module = osp.join(
                    self._cfg.OUTPUT_DIR,
                    "BUC_best_{:.3f}_{}_{}".format(self._state.current_rank1, self._state.current_step, module.__class__.__name__))
                module.save(path_to_best_module)
            self._state.best_rank1 = self._state.current_rank1
            self._state.best_mAP = self._state.current_mAP
            self._state.best_step = self._state.current_step
        self._logger.info("best mAP: {:.1%}, best rank1: {:.1%}".format(self._state.best_mAP, self._state.best_rank1))

    def _restore_checkpoint(self, module_list, suffix = None) -> None:
        for module in module_list:
            path_to_module = os.path.join(
                self._cfg.OUTPUT_DIR, "BUC_{}_{}".format(suffix, module.__class__.__name__))
            module.load(path_to_module)

    def restore_feature_extractor(self, path_to_weight):
        self._feature_extractor.load_state_dict(torch.load(path_to_weight))

    def _resume(self):
        self._logger.info("Resume training from step: {}".format(self._cfg.RESUME_STEP))
        self._restore_checkpoint(self._get_archive_modules(), self._cfg.RESUME_STEP)
        self._state.start_step = self._cfg.RESUME_STEP + 1
    
    def _fire_event_before_step(self):
        self._logger.info("========== step[{}] ==========".format(self._state.current_step))

    def _fire_event_after_step(self):
        for func, args, kwargs in self._events_after_step:
            func(*args, **kwargs)
    
    def get_purity(self):
        r""" Need to ensure that method '_reinitialize_settings' has been called before calculate purity of clusters. 
        So that all clusters would have been relabeled.
        """
        purity = self._disjoint_set.get_purity(self._databox.pids_train, self._databox.num_classes)
        self._logger.info("purity : {}".format(round(purity, 3)))
        return purity

    def add_handler_after_step(self, handler, *args, **kwargs):
        self._events_after_step.append((handler, args, kwargs))

    def _train_with_initial_settings(self, cfg):
        self._logger.info("---------- initial training ----------")
        model = self.pipe_for_train
        optimizer, scheduler = self._get_new_optimizer_and_scheduler(list(model.children()), [cfg.FEATURE_EXTRACTOR_LR, cfg.EMBEDDING_LAYER_LR])
        trainer = Trainor(model, optimizer, scheduler, self._disjoint_set)
        trainer.run(self._databox.train_loader, self._repell_loss, max_epochs=cfg.MAX_EPOCHS)
            
    def _bottom_up_clustering(self):
        self._logger.info("---------- bottom up clustering ----------")
        clustering(self._databox, self._NUM_TO_MERGE, self._disjoint_set, self.pipe_for_train)
        self._reinitialize_settings()

    def _reinitialize_settings(self):
        self._logger.info("Reinitialize lookup table")
        self._repell_loss.create_lookup_table(self.num_clusters)
        self._logger.info("Relabel clusters")
        self._disjoint_set.update_lable_map()

    def _get_new_optimizer_and_scheduler(self, modules, learning_rates):
        param_groups = []
        for module, lr in zip(modules, learning_rates):
            param_group = {'params': filter(lambda p: p.requires_grad, module.parameters()), 'lr': lr}
            param_groups.append(param_group)
        optimizer = torch.optim.SGD(param_groups, lr=0, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = MultiStepLR(optimizer, [15, ])
        return optimizer, scheduler
  
    def _train_with_new_settings(self):
        self._logger.info("---------- training ----------")
        model = torch.nn.Sequential(self._feature_extractor, self._embedding_layer)
        optimizer, scheduler = self._get_new_optimizer_and_scheduler(list(model.children()), [0.01, 0.001])
        trainer = Trainor(model, optimizer, scheduler, self._disjoint_set)
        trainer.run(self._databox.train_loader, self._repell_loss, max_epochs=2)

    def train(self, max_steps):
        for self._state.current_step in range(self._state.start_step, max_steps):
            self._fire_event_before_step()
            if self._state.current_step == -1:
                self._train_with_initial_settings(self._cfg.initial_node)
            else:
                self._bottom_up_clustering()
                self._train_with_new_settings()
            self._fire_event_after_step()

    def evaluate(self):
        self._logger.info("evaluating")
        evaluator = Evaluator(self._feature_extractor, R1_mAP(self._databox.num_query))
        cmc, mAP = evaluator.run(self._databox.val_loader)
        self._state.current_rank1 = cmc[0]
        self._state.current_mAP = mAP
        self._logger.info("mAP: {:.1%}, rank1: {:.1%}".format(mAP, cmc[0]))
    
    @staticmethod
    def main(cfg):
        pipeline_for_buc = Pipeline(cfg)
        if cfg.TRAIN:
            pipeline_for_buc.train(max_steps=int(1/cfg.CLUSTER_MERGE_PERCENT-1))
        else:
            pipeline_for_buc.restore_feature_extractor(cfg.TEST.WEIGHT)
            pipeline_for_buc.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.config_file != "":
        config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)
    config.freeze()

    output_dir = config.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("BUC", output_dir, 0)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info("Running with config:\n{}".format(config))

    os.environ['CUDA_VISIBLE_DEVICES'] = config.MODEL.DEVICE_ID
    cudnn.benchmark = True

    Pipeline.main(config)


