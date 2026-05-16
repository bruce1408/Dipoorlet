import os, sys
import subprocess
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()


def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    work_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_int8_120"
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_int8_120"
    input_txt = "/home/bruce_ultra/workspace/Quantization_Optimization/Quantizer-Tools/_outputs/dipoorlet_log/3_dipoorlet_models_resnet18/qnn_resnet18_calib_data_1.txt"
    os.makedirs(log_dir, exist_ok=True)
    
    
    # 构建 torchrun 命令
    command = [
        "qnn-net-run",
        "--backend", "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/libQnnHtp.so",
        "--retrieve_context", f"{work_dir}/qnn_resnet18_quant_basic.context.bin",
        "--input_list", input_txt,
        "--output_dir", log_dir
    ]
    
    # 执行命令
    subprocess.run(command, check=True)

    
if __name__ == "__main__":
    main()


