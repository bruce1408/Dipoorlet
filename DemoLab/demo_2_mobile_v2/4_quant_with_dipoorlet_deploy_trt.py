import os
import subprocess

def main():
    # 获取当前脚本所在的目录
    workdir = os.path.dirname(os.path.realpath(__file__))

    # 切换到该目录
    os.chdir(workdir)

    # 创建日志目录
    log_dir = os.path.join(workdir, "log_dir")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 构建 CUDA 环境变量
    cuda_visible_devices = "4,5,6,7"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    # 构建 torchrun 命令
    command = [
        "torchrun",
        "--nproc_per_node=4",
        "-m", "dipoorlet",
        "-M", "/mnt/share_disk/bruce_trie/Quantizer-Tools/Dipoorlet/DemoLab/demo_2_mobile_v2/models/mobilev2_model_new.onnx",
        "-I", "/mnt/share_disk/bruce_trie/Quantizer-Tools/Dipoorlet/DemoLab/demo_2_mobile_v2/trt/calibration_data",
        "-N", "1000",
        "-A", "mse",
        "-D", "trt",
        "-O", "./trt_dipoorlet_mse_log_dir"
    ]

    # 执行命令
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
