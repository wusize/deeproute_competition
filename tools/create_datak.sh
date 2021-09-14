#!/usr/bin/env bash

set -x
export PYTHONPATH=`pwd`:$PYTHONPATH

PARTITION=Pose
JOB_NAME=database
CONFIG=configs/benchmark/hv_centerpoint_secfpn_2x8_80e_pcdet_deeproute-9class.py
WORK_DIR=database
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SRUN_ARGS=${SRUN_ARGS:-""}
JOB_NAME=create_data

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/create_data.py kitti \
            --root-path /mnt/lustre/datatag/bixueting/KITTI \
            --out-dir /mnt/lustre/datatag/bixueting/KITTI \
            --extra-tag kitti
