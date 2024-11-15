import os
import subprocess

# 构建 CUDA 环境变量
cuda_visible_devices = "3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

def main():
    # 获取当前脚本所在的目录
    workdir = os.path.dirname(os.path.realpath(__file__))

    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{workdir}/trt_mobile_v2_dipoorlet_minmax"
    os.makedirs(os.path.join(workdir, log_dir), exist_ok=True)
    
    calibration_data = f"{workdir}/calibration_data"
    onnx_path = f"{workdir}/models/mobilev2_model_new.onnx"
    
    # 切换到该目录
    os.chdir(workdir)    

    # 构建 torchrun 命令
    command = [
        "torchrun",
        "--nproc_per_node=5",
        "-m", "dipoorlet",
        "-M", f"{onnx_path}",
        "-I", f"{calibration_data}",
        "-O", f"{log_dir}",
        "-N", "1000",
        "-A", "minmax",
        "-D", "trt"
    ]

    # 执行命令
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
