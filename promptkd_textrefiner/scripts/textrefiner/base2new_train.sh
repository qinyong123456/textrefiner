#!/bin/bash

# 检查是否提供了数据集参数
if [ $# -eq 0 ]; then
    echo "请指定数据集名称作为参数"
    echo "可用数据集: imagenet, caltech101, dtd, eurosat, fgvc_aircraft, oxford_flowers, food101, oxford_pets, sun397, ucf101"
    exit 1
fi

# custom config
DATA="/kaggle/working/textrefiner/data"
TRAINER=TextRefiner

# 通过命令行参数获取数据集
DATASET=$1
SEED=1

CFG=vit_b16_c2_ep20_batch8_4+4ctx
SHOTS=16

DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}

# 根据不同数据集设置合适的KD_WEIGHT
case $DATASET in
    fgvc_aircraft|oxford_flowers|dtd)
        KD_WEIGHT=200.0
        ;;
    *)
        KD_WEIGHT=1000.0
        ;;
esac

CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.MODAL base2novel \
    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
    TRAINER.PROMPTKD.KD_WEIGHT ${KD_WEIGHT}
