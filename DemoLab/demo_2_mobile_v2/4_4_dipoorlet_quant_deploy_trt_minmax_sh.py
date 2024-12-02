import os, sys
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import demo_utils.quant_config as config

# 构建 CUDA 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_ids
cuda_nums = len(config.cuda_ids.split(","))

def main():
    
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{config.export_work_dir}/trt_mobile_v2_dipoorlet_minmax"
    os.makedirs(log_dir, exist_ok=True)
    
    calibration_data = config.dipoorlet_calib_dir
    onnx_path = f"{config.export_work_dir}/mobilev2_model_new.onnx"   

    # 构建 torchrun 命令
    command = [
        "torchrun",
        f"--nproc_per_node={cuda_nums}",
        "-m", "dipoorlet",
        "-M", onnx_path,
        "-I", calibration_data,
        "-O", log_dir,
        "-N", "10",
        "-A", "minmax",
        "-D", "trt",
        "--onnx_sim"
    ]

    # 执行命令
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
