import os, sys, datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import demo_utils.quant_config as config
import subprocess

# 构建 CUDA 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_ids
cuda_nums = len(config.cuda_ids.split(","))


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
algorithm = "brecq"

def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{config.od_bev_outputs}/snpe_od_bev_dipoorlet_{algorithm}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    onnx_path = f"{config.od_bev_onnx_models}/od_bev_1110.onnx"
    
    # 构建 torchrun 命令
    command = [
        "torchrun",
        f"--nproc_per_node={cuda_nums}",
        "-m", "dipoorlet",
        "-M", onnx_path,
        "-I", config.od_bev_calibration_data_dipoorlet,
        "-O", log_dir,
        "-N", "24",
        "-A", "mse",
        "-D", "snpe",
        f"--{algorithm}"
    ]

        # 执行命令
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
