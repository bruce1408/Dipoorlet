import os, sys
import subprocess
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()


def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_int8_100"
    
    # log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_int8_100_20250927_1832_cle_aimet"


    
    # 构建 torchrun 命令
    command = [
        "qnn-model-lib-generator",
        "-c", f"{log_dir}/qnn_resnet18_quant_basic.cpp",
        "-b", f"{log_dir}/qnn_resnet18_quant_basic.bin",
        "-t", "x86_64-linux-clang",
        "-o", log_dir,
        # "-l", "qnn_resnet18_quant_fp16",
    ]
    
    # 执行命令
    subprocess.run(command, check=True)

    
if __name__ == "__main__":
    main()


