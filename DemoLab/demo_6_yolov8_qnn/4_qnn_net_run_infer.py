import os, sys
import subprocess
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    # log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_int8_1000_20250920_215500"
    log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_fp16_20250921_010631"

    # 构建 torchrun 命令
    command = [
        "qnn-net-run",
        "--backend", "/share/qnn-helper/sdk/qaisw-v2.26.0.250121145233_87812-auto/lib/x86_64-linux-clang/libQnnHtp.so",
        "--retrieve_context", f"{log_dir}/qnn_yolov8_quant_basic.context.bin",
        "--input_list", "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/dipoorlet_log/4_dipoorlet_models_yolov8/qnn_single_pic.txt",
        "--output_dir", log_dir
    ]
    
    # 执行命令
    subprocess.run(command, check=True)

    
if __name__ == "__main__":
    main()


