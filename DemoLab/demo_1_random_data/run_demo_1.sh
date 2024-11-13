#!/usr/bin/env bash

# 获取当前脚本所在的目录
workdir=$(cd "$(dirname "$0")" || exit; pwd)

# 切换到该目录
cd "${workdir}" || exit

if [ ! -d "log_dir" ]; then
    mkdir log_dir
fi

torchrun \
-m dipoorlet \
-M /mnt/share_disk/bruce_trie/onnx_models/resnet34_model.onnx \
-I /mnt/share_disk/bruce_trie/Quantizer-Tools/Dipoorlet/DemoLab/demo_1_random_data \
-N 2 \
-A minmax \
-D trt \
-O ./log_dir \
--adaround \
--ada_epoch 1  