import os, sys
import subprocess
# import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
from calculate_trt_engine_acc import calculate_tensorrt_acc
cfg = get_cfg_defaults()

# 构建 CUDA 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SYSTEM.CUDA_IDS
os.environ["OMP_NUM_THREADS"] = cfg.DIPOORLET.OMP_NUM_THREADS  # 设置OpenMP线程数，可以根据CPU核心数调整
cuda_num = len(cfg.SYSTEM.CUDA_IDS.split(","))

def main():
    
    log_dir = f"{cfg.DIPOORLET.tensorrt_export_dir}/trt_resnet18_hist"
    os.makedirs(log_dir, exist_ok=True)
    
    onnx_path = f"{cfg.SYSTEM.MODELS_DIR}/resnet18.onnx"
    calibration_data = f"{cfg.DIPOORLET.dipoorlet_calib_data_dir}/resnet18_calib/" 

    # 构建 torchrun 命令
    command = [
        "torchrun",
        f"--nproc_per_node={cuda_num}",
        "--master_port=29501",
        "-m", "dipoorlet",
        "-M", onnx_path,
        "-I", calibration_data,
        "-O", log_dir,
        "-N", "100",
        "-A", "hist",
        "--onnx_sim",
        "-D", "trt"
    ]
    
    # 执行命令
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
    # calculate_tensorrt_acc(engine_file=f"{log_dir}/resnet18_model_dipoorlet_hist_int8.engine")
