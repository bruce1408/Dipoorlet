import os, sys
import subprocess
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

# 构建 CUDA 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SYSTEM.CUDA_IDS
os.environ["OMP_NUM_THREADS"] = cfg.DIPOORLET.OMP_NUM_THREADS  # 设置OpenMP线程数，可以根据CPU核心数调整
cuda_nums = len(cfg.SYSTEM.CUDA_IDS.split(","))


def main():
    # 命名规则按照 = 量化工具+平台+模型+量化算法
    log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_int8_500_{cfg.SYSTEM.TIMESTAMP}"
    os.makedirs(log_dir, exist_ok=True)

    calib_data_txt = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_calib_data_500.txt"
    
    onnx_path = cfg.DIPOORLET.yolov8_onnx_models

    # 构建 torchrun 命令
    command = [
        "torchrun",
        f"--nproc_per_node={cuda_nums}",
        "--master_port=29501",
        "-m", "dipoorlet",
        "-M", onnx_path,
        "-I", cfg.DIPOORLET.dipoorlet_calib_data_dir,
        "-O", log_dir,
        "-N", "500",
        "-A", "mse",
        "--onnx_sim",
        "-D", "snpe"
    ]

        # 执行命令
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
