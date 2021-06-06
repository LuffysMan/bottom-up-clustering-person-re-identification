workspace="your workspace to contain datasets and logs"
data="$workspace/Datasets/Reid"
outdir="$workspace/Log/market1501/method_BUC/z_end2end_default_settings"
weight="$outdir/model_best.pth"

python3 method_BUC/pipeline.py \
 OUTPUT_DIR $outdir \
 TRAIN 1 \
 RESUME 0 \
 MODEL.DEVICE_ID "('0')" \
 DATASETS.ROOT_DIR $data \
 DATASETS.NAMES "('market1501','market1501')" \
 DATASETS.TARGET "market1501" \
 DATALOADER.IMS_PER_BATCH 16 \
 TEST.WEIGHT $weight 