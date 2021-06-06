import torch
import logging

from libreid.utils.iotools import get_time_cost


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainor(object):
    def __init__(self, model, optimizer, scheduler, Y):
        self._logger = logging.getLogger("BUC")
        self.__model = model
        self.__optimizer = optimizer
        self.__scheduler = scheduler
        self.__Y = Y

    def run(self, data, loss_fn, max_epochs=0):
        self.__model.cuda()
        self.__model.train()
        for epoch in range(max_epochs):
            self.run_once(data, loss_fn, epoch)
            self.__scheduler.step()

    @get_time_cost
    def run_once(self, data, loss_fn, epoch):
        lossAverageMeter = AverageMeter()
        accAverageMeter = AverageMeter()
        for iteration, batch in enumerate(data):
            imgs, targets, *_, sample_ids = batch
            imgs = imgs.to("cuda:0")
            targets = self.__Y.get_labels(sample_ids)
            feats = self.__model(imgs)
            loss, scores = loss_fn(feats, targets)
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            lossAverageMeter.update(loss)
            pseudo_acc = (scores.max(1)[1] == targets).float().mean()
            accAverageMeter.update(pseudo_acc)
            if iteration % 200 == 0:
                self._logger.info("Epoch[{}|{}]\t"
                    "loss: {:.3f} ({:.3f})\t"
                    "ps_acc: {:.1%} ({:.1%})\t"
                    "lr: {:.3E}".format(
                    epoch, iteration,
                    lossAverageMeter.val.item(), lossAverageMeter.avg.item(), 
                    accAverageMeter.val.item(), accAverageMeter.avg.item(),
                    self.__scheduler.get_last_lr()[0]))

class Evaluator(object):
    def __init__(self, model, metric):
        self.__model = model
        self.__metric = metric

    def run(self, data):
        self.__model.cuda()
        self.__model.eval()
        with torch.no_grad():
            for batch in data:
                imgs, pids, camids, *_ = batch
                imgs = imgs.to("cuda:0")
                feats = self.__model(imgs)
                self.__metric.update((feats, pids, camids))
            cmc, mAP = self.__metric.compute()
            return cmc, mAP
