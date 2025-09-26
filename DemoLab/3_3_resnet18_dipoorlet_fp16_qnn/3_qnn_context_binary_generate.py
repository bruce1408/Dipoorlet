import os, sys
import subprocess
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_int8_100_20250924_113605"
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_quant_fp16_20250925_1930"
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_quant_fp16_20250926_1732"
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_quant_fp16_20250926_2149"



    
    # 构建 torchrun 命令
    command = [
        "qnn-context-binary-generator",
        "--backend", "/share/qnn-helper/sdk/qaisw-v2.26.0.250121145233_87812-auto/lib/x86_64-linux-clang/libQnnHtp.so",
        "--model", f"{log_dir}/x86_64-linux-clang/libqnn_resnet18_quant_fp16.so",
        "--binary_file", f"{log_dir}/qnn_resnet18_quant_fp16.context",
        "--output_dir", log_dir,
        "--config_file", "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/Dipoorlet/DemoLab/3_3_resnet18_dipoorlet_fp16_qnn/be_htp.json"
    ]
    
    # 执行命令
    subprocess.run(command, check=True)

    
if __name__ == "__main__":
    main()


