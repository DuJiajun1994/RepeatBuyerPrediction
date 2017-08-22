#!/bin/bash
# Usage:
# bash scripts/test.sh GPU_ID MODEL DATA CONFIG
#
# Example:
# bash scripts/test.sh 0 data1_dnn data1 cfg1

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
MODEL=$2
DATA=$3
CONFIG=$4

LOG="logs/repeat_buyer_prediction_${MODEL}_`date +'%Y_%m_%d_%H_%M_%S'`.txt"
exec &> >(tee -a "$LOG")
echo "Logging output to ${LOG}"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
time python lib/test.py --model ${MODEL} --data ${DATA} --cfg ${CONFIG}
