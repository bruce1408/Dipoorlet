import os, sys
import subprocess
import time, datetime
from common.configs import get_cfg_defaults
from quant_tools.common_utils import time_it
cfg = get_cfg_defaults()

# 构建 CUDA 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SYSTEM.CUDA_IDS
os.environ["OMP_NUM_THREADS"] = cfg.DIPOORLET.OMP_NUM_THREADS  # 设置OpenMP线程数，可以根据CPU核心数调整
cuda_nums = len(cfg.SYSTEM.CUDA_IDS.split(","))


@time_it
def main():
    
    log_dir = f"{cfg.DIPOORLET.tensorrt_export_dir}/trt_resnet18_adaround"
    os.makedirs(log_dir, exist_ok=True)
    
    onnx_path = f"{cfg.SYSTEM.MODELS_DIR}/resnet18.onnx"
    calibration_data = f"{cfg.DIPOORLET.dipoorlet_calib_data_dir}/resnet18_calib/"    
    
    command = [
        "torchrun",
        "--master_port=29503",
        f"--nproc_per_node={cuda_nums}",
        "-m", "dipoorlet",
        "-M", onnx_path,
        "-I", calibration_data,
        "-O", log_dir,
        "-N", "100",
        "-A", "mse",
        "-D", "trt",
        "--onnx_sim",
        "--adaround"
    ]
    
    # 执行命令
    subprocess.run(command, check=True)
    
if __name__ == "__main__":
    main()
