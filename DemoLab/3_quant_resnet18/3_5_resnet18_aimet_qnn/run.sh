#!/usr/bin/env bash

# 获取当前脚本所在的目录
workdir=$(cd "$(dirname "$0")" || exit; pwd)

# 切换到该目录
cd "${workdir}" || exit

if [ ! -d "log_dir" ]; then
    mkdir log_dir
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
--nproc_per_node=4 \
-m dipoorlet \
-M /mnt/share_disk/bruce_trie/Quantizer-Tools/Dipoorlet/DemoLab/demo_2_mobile_v2/models/mobilev2_model_new.onnx \
-I /mnt/share_disk/bruce_trie/Quantizer-Tools/Dipoorlet/DemoLab/demo_2_mobile_v2/trt/calibration_data \
-N 1000 \
-A mse \
-D trt \
-O ./trt_dipoorlet_mse_log_dir 