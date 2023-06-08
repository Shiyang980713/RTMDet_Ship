#!/usr/bin/env zsh

GPU_IDs=4
CONFIG_FILE='/remote-home/syfeng/MyProject/mmrotate/configs/rotated_rtmdet/rotated_rtmdet_l-coco_pretrain-3x-ship_ms.py'
CHECKPOINT='/remote-home/syfeng/MyProject/mmrotate/work_dirs/rotated_rtmdet_l-coco_pretrain-3x-ship_ms/epoch_36.pth'

NUM_GPUS=${#GPU_IDs//,/}
#dist train shell
CUDA_VISIBLE_DEVICES=${GPU_IDs} PORT=22200 \
./tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} \
${NUM_GPUS} \
--work-dir work_dirs_debug/debug_ms_36 \
