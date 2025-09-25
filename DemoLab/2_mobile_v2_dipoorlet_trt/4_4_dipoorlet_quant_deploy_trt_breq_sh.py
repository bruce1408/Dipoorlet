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
    log_dir = f"{cfg.DIPOORLET.tensorrt_export_dir}/trt_mobile_v2_dipoorlet_brecq_{cfg.SYSTEM.TIMESTAMP}"
    os.makedirs(log_dir, exist_ok=True)

    calibration_data = cfg.DIPOORLET.dipoorlet_calib_data_dir
    onnx_path = f"{cfg.SYSTEM.MODELS_DIR}/mobilev2_model_trained.onnx"   

    
    # 构建 torchrun 命令
    command = [
        "torchrun",
        "--master_port=29502",
        f"--nproc_per_node={cuda_nums}",
        "-m", "dipoorlet",
        "-M", onnx_path,
        "-I", calibration_data,
        "-O", log_dir,
        "-N", "100",
        "-A", "mse",
        "-D", "trt",
        "--onnx_sim",
        "--brecq"
    ]
    
    # 执行命令
    subprocess.run(command, check=True)

    
if __name__ == "__main__":
    main()
