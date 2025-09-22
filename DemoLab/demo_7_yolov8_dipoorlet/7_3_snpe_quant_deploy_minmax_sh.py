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
    log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/dipoorlet_yolov8_quant_int8_500_minmax_{cfg.SYSTEM.TIMESTAMP}"
    os.makedirs(log_dir, exist_ok=True)
    
    onnx_path = cfg.DIPOORLET.yolov8_onnx_models

    # 构建 torchrun 命令
    command = [
        "torchrun",
        f"--nproc_per_node={cuda_nums}",
        "--master_port=29502",
        "-m", "dipoorlet",
        "-M", onnx_path,
        "-I", f"{cfg.DIPOORLET.dipoorlet_calib_data_dir}/yolov8_calib/",
        "-O", log_dir,
        "-N", "100",
        "-A", "minmax",
        "-D", "snpe",
        "--onnx_sim"
    ]

    # 执行命令
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
