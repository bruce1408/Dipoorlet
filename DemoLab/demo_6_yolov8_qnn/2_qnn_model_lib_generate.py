import os, sys
import subprocess
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()


def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    # log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_int8_1000_20250920_215500"
    # log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_fp16_20250921_010631"
    log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_mixed_20250921_190437"
    
    
    # 构建 torchrun 命令
    command = [
        "qnn-model-lib-generator",
        "-c", f"{log_dir}/qnn_yolov8_quant_basic.cpp",
        "-b", f"{log_dir}/qnn_yolov8_quant_basic.bin",
        "-t", "x86_64-linux-clang",
        "-o", log_dir,
    ]
    
    # 执行命令
    subprocess.run(command, check=True)

    
if __name__ == "__main__":
    main()


