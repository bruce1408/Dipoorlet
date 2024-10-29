#!/usr/bin/env bash

torchrun \
-m dipoorlet \
-M /mnt/share_disk/bruce_trie/misc_data_products/onnx_models/resnet34_model.onnx \
-I /mnt/share_disk/bruce_trie/Quantizer-Tools/Dipoorlet/example/demo_1_classify/ \
-N 2 \
-A minmax \
-D trt


