import os, sys
import subprocess
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_quant_mixed_{cfg.SYSTEM.TIMESTAMP}"
    calib_data_txt = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_calib_data_500.txt"
    onnx_path = cfg.DIPOORLET.yolov8_onnx_models
    
    os.makedirs(log_dir, exist_ok=True)
    overrides_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/inferdet/tools/yolov8_overrides_mixed.json"
    
    command = [
        "qnn-onnx-converter",
        "--input_network", onnx_path,
        "--input_list", calib_data_txt,
        "-o", f"{log_dir}/qnn_yolov8_quant_basic.cpp",
        "--use_per_channel_quantization",
        "--quantization_overrides", overrides_path
        # --act_bitwidth 8 --bias_bitwidth 32 --weights_bitwidth 8
    ]
    
    # 执行命令
    subprocess.run(command, check=True)

    
if __name__ == "__main__":
    main()
