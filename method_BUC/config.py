from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

BUC = CN()

# ---------------------------------------------------------------------------- #
# Options for whole pipleline of BUC
# ---------------------------------------------------------------------------- #
BUC.OUTPUT_DIR = ""              # Path to checkpoint and saved log of trained model
BUC.TRAIN = 0                    # 0: evaluating, 1: training
BUC.RESUME = 0                   # 0: train from scratch, 1: resume from a checkpoint. Only become valid when TRAIN is set to 1. 
BUC.RESUME_STEP = -2             # Only become valid when IF_RESUME is set to 1.

# Settings for clustering process 
BUC.CLUSTER_MERGE_PERCENT = 0.05     
BUC.CLUSTER_DIVERSITY_WEIGHT = 0.005

BUC.initial_node = CN()
BUC.initial_node.FEATURE_EXTRACTOR_LR = 0.01
BUC.initial_node.EMBEDDING_LAYER_LR = 0.1
BUC.initial_node.MAX_EPOCHS = 20

BUC.MODEL = CN()
# Using cuda or cpu for training
BUC.MODEL.DEVICE = "cuda"
# ID number of GPU
BUC.MODEL.DEVICE_ID = '0'
# Name of backbone
BUC.MODEL.NAME = 'resnet50'
# Last stride of backbone
BUC.MODEL.PRETRAIN_PATH = "/home/luffy/Workspace/Models/resnet50-19c8e357.pth"
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
BUC.MODEL.PRETRAIN_CHOICE = 'imagenet'

BUC.MODEL.DROPOUT_RATE = 0.5
# normalize feature
BUC.MODEL.IF_WITH_NORMALIZATION = 'no'
# froze resnet BN modules in (bn1, layer1, layer2)
BUC.MODEL.FROZE_BN_LAYERS = 'no'

BUC.MODEL.FEAT_DIM = 1024


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
BUC.INPUT = CN()
# Size of the image during training
BUC.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
BUC.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
BUC.INPUT.FLIP_PROB = 0.5
# Random probability for random erasing
BUC.INPUT.RE_PROB = 0.0
# Values to be used for image normalization
BUC.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
BUC.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
BUC.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
BUC.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
BUC.DATASETS.NAMES = ('market1501','market1501')
BUC.DATASETS.TARGET = 'market1501'
# Root directory where datasets should be used (and downloaded if not found)
BUC.DATASETS.ROOT_DIR = ('~/Workspace/Datasets/Reid')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
BUC.DATALOADER = CN()
# Number of data loading threads
BUC.DATALOADER.NUM_WORKERS = 2
# Sampler for data loading
BUC.DATALOADER.SAMPLER = 'softmax'
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
BUC.DATALOADER.IMS_PER_BATCH = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
BUC.SOLVER = CN()
# Name of optimizer
BUC.SOLVER.OPTIMIZER_NAME = "SGD"
# Base learning rate
BUC.SOLVER.BASE_LR = 0.1
BUC.SOLVER.BIAS_LR_FACTOR = 2
# Momentum
BUC.SOLVER.MOMENTUM = 0.9
# Settings of weight decay
BUC.SOLVER.WEIGHT_DECAY = 0.0005
BUC.SOLVER.WEIGHT_DECAY_BIAS = 0.0
# decay rate of learning rate
BUC.SOLVER.GAMMA = 0.1

# Number of max epoches
BUC.SOLVER.MAX_EPOCHS = 20
# decay step of learning rate
BUC.SOLVER.STEPS = (15,)
# epoch number of saving checkpoints
BUC.SOLVER.CHECKPOINT_PERIOD = 20
# epoch number of validation
BUC.SOLVER.EVAL_PERIOD = 20
# iteration of display training log
BUC.SOLVER.LOG_PERIOD = 100






# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
BUC.TEST = CN()
# Number of images per batch during test
BUC.TEST.IMS_PER_BATCH = 128
# Path to trained model
BUC.TEST.WEIGHT = ""
