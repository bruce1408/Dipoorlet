import os, sys
import subprocess
import DemoLab.demo_utils.quant_config as config


# 构建 CUDA 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_ids
os.environ["OMP_NUM_THREADS"] = config.OMP_NUM_THREADS  # 设置OpenMP线程数，可以根据CPU核心数调整
cuda_nums = len(config.cuda_ids.split(","))

def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{config.export_work_dir}/trt_mobile_v2_dipoorlet_mse"
    os.makedirs(log_dir, exist_ok=True)
    
    onnx_path = f"{config.export_work_dir}/mobilev2_model_new.onnx"
    
    # 构建 torchrun 命令
    command = [
        "torchrun",
        f"--nproc_per_node={cuda_nums}",
        "-m", "dipoorlet",
        "-M", onnx_path,
        "-I", config.dipoorlet_calib_dir,
        "-O", log_dir,
        "-N", "10",
        "-A", "mse",
        "--onnx_sim",
        "-D", "trt"
    ]

        # 执行命令
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
