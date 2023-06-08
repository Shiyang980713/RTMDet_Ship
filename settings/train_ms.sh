#!/usr/bin/env zsh

GPU_IDs=0,1
# CONFIG_FILE='/remote-home/syfeng/MyProject/mmrotate/configs/rotated_rtmdet/rotated_rtmdet_l-3x-ship.py'
CONFIG_FILE='/remote-home/syfeng/MyProject/mmrotate/configs/rotated_rtmdet/rotated_rtmdet_l-coco_pretrain-3x-ship_ms.py'

NUM_GPUS=${#GPU_IDs//,/}
#dist train shell
CUDA_VISIBLE_DEVICES=${GPU_IDs} PORT=28200 \
./tools/dist_train.sh \
${CONFIG_FILE} \
${NUM_GPUS} \
--resume \
