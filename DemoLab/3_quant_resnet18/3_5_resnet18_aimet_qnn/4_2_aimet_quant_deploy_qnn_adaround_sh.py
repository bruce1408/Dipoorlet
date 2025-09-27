import os, sys
import subprocess
from common.configs import get_cfg_defaults
from spectrautils.print_utils import *
cfg = get_cfg_defaults()

# 构建 CUDA 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SYSTEM.CUDA_IDS
os.environ["OMP_NUM_THREADS"] = cfg.DIPOORLET.OMP_NUM_THREADS  # 设置OpenMP线程数，可以根据CPU核心数调整
cuda_nums = len(cfg.SYSTEM.CUDA_IDS.split(","))

def dipoorlet_quant_deploy(mode="trt"):
    
    print_colored_text(f"Start dipoorlet-quant-deploy, Please wait a moment...", "green")
    # 构建 torchrun 命令
    command = [
        "torchrun",
        f"--nproc_per_node={cuda_nums}",
        "--master_port=29501",
        "-m", "dipoorlet",
        "-M", onnx_path,
        "-I", calibration_data,
        "-O", log_dir,
        "-N", "100",
        "-A", "mse",
        "--onnx_sim",
        "-D", f"{mode}"
    ]
    
    # 执行命令
    subprocess.run(command, check=True)
    
    

def qnn_onnx_converter(mode="snpe"):
    
    print_colored_text(f"Start qnn-onnx-converter, Please wait a moment...", "green")
    # 构建 torchrun 命令
    command = [
        "qnn-onnx-converter",
        "--input_network", onnx_path,
        "--input_list", calib_data_txt,
        "-o", f"{log_dir}/qnn_resnet18_quant_basic.cpp",
        "--quantization_overrides", overrides_path,
        # "--use_per_channel_quantization",
        # --act_bitwidth 8 --bias_bitwidth 32 --weights_bitwidth 8
    ]
    
    subprocess.run(command, check=True)

def qnn_model_lib_generator(mode="snpe"):
    print_colored_text(f"Start qnn-model-lib-generator, Please wait a moment...", "green")
    
    command = [
        "qnn-model-lib-generator",
        "-c", f"{log_dir}/qnn_resnet18_quant_basic.cpp",
        "-b", f"{log_dir}/qnn_resnet18_quant_basic.bin",
        "-t", "x86_64-linux-clang",
        "-o", log_dir,
    ]
    
    subprocess.run(command, check=True)


def qnn_context_binary_generator(mode="snpe"):
    
    print_colored_text(f"Start qnn-context-binary-generator, Please wait a moment...", "green")
    command = [
        "qnn-context-binary-generator",
        "--backend", "/share/qnn-helper/sdk/qaisw-v2.26.0.250121145233_87812-auto/lib/x86_64-linux-clang/libQnnHtp.so",
        "--model", f"{log_dir}/x86_64-linux-clang/libqnn_resnet18_quant_basic.so",
        "--binary_file", f"{log_dir}/qnn_resnet18_quant_basic.context",
        "--output_dir", log_dir
    ]
    
    subprocess.run(command, check=True)


if __name__ == "__main__":
    
    MODE="snpe"
    
    calibration_data = f"{cfg.DIPOORLET.dipoorlet_calib_data_dir}/resnet18_calib/"
    calib_data_txt = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_calib_data_1.txt"
    
    # =================== aimet adaround ===================
    onnx_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/aimet_log/resnet18_adaround/adaround_2025-09-27-18-59-03/adaround_resnet.onnx"
    overrides_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/aimet_log/resnet18_adaround/adaround_2025-09-27-18-59-03/adaround_resnet_simplified.json"
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/{MODE}_dipoorlet_resnet18_aimet_adaround"
    os.makedirs(log_dir, exist_ok=True)
    # =================== aimet adaround ===================
    

    qnn_onnx_converter(MODE)
    
    qnn_model_lib_generator(MODE)
    
    qnn_context_binary_generator(MODE)
    
    
    
    