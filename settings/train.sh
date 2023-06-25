#!/usr/bin/env zsh

GPU_IDs=0,1
# CONFIG_FILE='/remote-home/syfeng/MyProject/ship_det/RTMDet_Ship/configs/rotated_rtmdet/rotated_rtmdet_lsk-3x-ship.py'
# CONFIG_FILE='/remote-home/syfeng/MyProject/ship_det/RTMDet_Ship/configs/rotated_rtmdet/rotated_rtmdet_lsk_pre_8e-3-3x-ship.py'
# CONFIG_FILE='/remote-home/syfeng/MyProject/ship_det/RTMDet_Ship/configs/rotated_rtmdet/rotated_rtmdet_lsk_pre_fpn-1x-ship.py'
CONFIG_FILE='/remote-home/syfeng/MyProject/ship_det/RTMDet_Ship/configs/rotated_rtmdet/rotated_rtmdet_lsk_pre_fpn5-3x-ship.py'

NUM_GPUS=${#GPU_IDs//,/}
#dist train shell
CUDA_VISIBLE_DEVICES=${GPU_IDs} PORT=28200 \
./tools/dist_train.sh \
${CONFIG_FILE} \
${NUM_GPUS} \
