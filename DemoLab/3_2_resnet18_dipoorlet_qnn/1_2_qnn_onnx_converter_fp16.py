import os, sys
import subprocess
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_quant_fp16_20250925_1930"
    os.makedirs(log_dir, exist_ok=True)
    onnx_path = f"{cfg.SYSTEM.MODELS_DIR}/resnet18.onnx"

    
    # 构建 torchrun 命令
    command = [
        "qnn-onnx-converter",
        "--input_network", onnx_path,
        "--float_bitwidth", "16",
        "-o", f"{log_dir}/qnn_resnet18_quant_fp16.cpp"
    ]
    
    # 执行命令
    subprocess.run(command, check=True)

    
if __name__ == "__main__":
    main()
