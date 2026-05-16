import os, sys
import subprocess
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_int8_120"
    
    # 构建 torchrun 命令
    command = [
        "qnn-context-binary-generator",
        "--backend", "libQnnHtp.so",
        "--model", f"{log_dir}/x86_64-linux-clang/libqnn_resnet18_quant_basic.so",
        "--binary_file", f"{log_dir}/qnn_resnet18_quant_basic.context",
        "--output_dir", log_dir
    ]
    
    # 执行命令
    subprocess.run(command, check=True)

    
if __name__ == "__main__":
    main()


