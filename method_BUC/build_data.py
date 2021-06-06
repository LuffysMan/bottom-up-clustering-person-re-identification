# encoding: utf-8
import random
import math
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image

from libreid.data.datasets import init_dataset
from .dataset_loader import ImageDataset


class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)


def _build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            # T.Resize(cfg.INPUT.SIZE_TRAIN),
            RandomSizedRectCrop(*cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB),
            # T.Pad(cfg.INPUT.PADDING),
            # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform


def train_collate_fn(batch):
    imgs, pids, camids, _, img_ids = zip(*batch)
    pids = torch.as_tensor(pids, dtype=torch.int64)
    camids = torch.as_tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, list(img_ids)


def val_collate_fn(batch):
    imgs, pids, camids, _, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids


def make_data_loader(cfg):
    train_transforms = _build_transforms(cfg, is_train=True)
    val_transforms = _build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    def make_databox(name):
        class DataBox(object):
            pass

        dataset = init_dataset(name, root=cfg.DATASETS.ROOT_DIR)
        # dataset.train.sort(key=lambda x: (x[1], x[2], x[0]))
        train_set = ImageDataset(dataset.train, train_transforms)

        train_loader = DataLoader(
            train_set, batch_size=cfg.DATALOADER.IMS_PER_BATCH, shuffle=True, collate_fn=train_collate_fn, 
            num_workers=num_workers, pin_memory=True, drop_last=True )

        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
        val_loader = DataLoader(
            val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn )

        databox = DataBox()
        databox.train_loader = train_loader
        databox.num_samples = len(dataset.train)
        databox.num_classes = dataset.num_train_pids
        databox.num_cameras = dataset.num_train_cams
        databox.pids_train = [pid for (_, pid, _) in dataset.train]
        databox.val_loader = val_loader
        databox.num_query = len(dataset.query)

        return databox

    # names = cfg.DATASETS.NAMES.split(',')
    return make_databox(cfg.DATASETS.NAMES[1].strip())


def make_val_loader(cfg):
    val_transforms = _build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    # load target dataset
    val_dataset = init_dataset(cfg.DATASETS.TARGET, root=cfg.DATASETS.ROOT_DIR)

    val_set = ImageDataset(val_dataset.query +
                           val_dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return val_loader, len(val_dataset.query)
