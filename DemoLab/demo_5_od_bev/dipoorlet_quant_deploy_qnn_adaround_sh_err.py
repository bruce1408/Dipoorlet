import subprocess
import os, sys, datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dipoorlet_utils.quant_config as config

# 构建 CUDA 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_ids

# 设置OpenMP线程数，可以根据CPU核心数调整
os.environ["OMP_NUM_THREADS"] = config.OMP_NUM_THREADS   

cuda_nums = len(config.cuda_ids.split(","))

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
algorithm = "adaround"

def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{config.od_bev_outputs}/qnn_od_bev_dipoorlet_{algorithm}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    onnx_path = Path(f"{config.od_bev_onnx_models}/od_bev_1110.onnx")
    
    # 验证输入文件存在
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX模型文件不存在: {onnx_path}")
    
    # 构建 torchrun 命令
    command = [
        "torchrun",
        f"--nproc_per_node={cuda_nums}",
        "--master_port=29501",  # 使用新端口
        "-m", "dipoorlet",
        "-M", str(onnx_path),
        "-I", config.od_bev_calibration_data_dipoorlet,
        "-O", log_dir,
        "-N", "6",
        "-A", "mse",
        "-D", "snpe",
        "--onnx_sim",
        f"--{algorithm}"
    ]
    
    # 执行命令
    # subprocess.run(command, check=True)
    
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Standard Output: {e.stdout.decode()}")
        print(f"Standard Error: {e.stderr.decode()}")

if __name__ == "__main__":
    main()
