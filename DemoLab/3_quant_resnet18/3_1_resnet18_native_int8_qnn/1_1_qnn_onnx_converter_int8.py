import os, sys
import subprocess
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_int8_100_20250927_1832_cle_aimet"
    os.makedirs(log_dir, exist_ok=True)

    calib_data_txt = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_calib_data_100.txt"
    onnx_path = f"{cfg.SYSTEM.MODELS_DIR}/resnet18.onnx"
    onnx_path = f"{cfg.SYSTEM.MODELS_DIR}/resnet18_cle_aimet.onnx"

    
    # 构建 torchrun 命令
    command = [
        "qnn-onnx-converter",
        "--input_network", onnx_path,
        "--input_list", calib_data_txt,
        "-o", f"{log_dir}/qnn_resnet18_quant_basic.cpp",
        "--use_per_channel_quantization",
        # --act_bitwidth 8 --bias_bitwidth 32 --weights_bitwidth 8
    ]
    
    # 执行命令
    subprocess.run(command, check=True)

    
if __name__ == "__main__":
    main()
