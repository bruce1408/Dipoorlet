import os, sys
import subprocess
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

# 构建 CUDA 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SYSTEM.CUDA_IDS
os.environ["OMP_NUM_THREADS"] = cfg.DIPOORLET.OMP_NUM_THREADS  # 设置OpenMP线程数，可以根据CPU核心数调整
cuda_nums = len(cfg.SYSTEM.CUDA_IDS.split(","))


def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_basic_{cfg.SYSTEM.TIMESTAMP}"
    os.makedirs(log_dir, exist_ok=True)

    calib_data_txt = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_calib_data.txt"
    onnx_path = cfg.DIPOORLET.yolov8_onnx_models

    
    # 构建 torchrun 命令
    command = [
        "qnn-onnx-converter",
        "--input_network", onnx_path,
        "--input_list", calib_data_txt,
        "-o", f"{log_dir}/qnn_yolov8_quant_basic.cpp",
        "--use_per_channel_quantization",
        # "--quantization_overrides"
        # --act_bitwidth 8 --bias_bitwidth 32 --weights_bitwidth 8

    ]
    
    # 执行命令
    subprocess.run(command, check=True)

    
if __name__ == "__main__":
    main()
