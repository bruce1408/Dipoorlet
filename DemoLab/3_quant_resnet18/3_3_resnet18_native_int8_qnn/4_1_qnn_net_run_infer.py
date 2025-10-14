import os, sys
import subprocess
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_int8_100_20250924_113605"
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_quant_fp16_20250925_1930"
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_int8_100"

    
    
    # 构建 torchrun 命令
    command = [
        "qnn-net-run",
        "--backend", "libQnnHtp.so",
        "--retrieve_context", f"{log_dir}/qnn_resnet18_quant_basic.context.bin",
        "--input_list", f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_raw_data_full.txt",
        "--output_dir", f"{log_dir}/qnn_resnet18_quant_fp16_infer_res_full"
    ]

    # 执行命令
    subprocess.run(command, check=True)

    
if __name__ == "__main__":
    main()


