import os, sys
import subprocess
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

def main():
    # 命名规则按照 = 平台+模型+量化工具+量化算法
    log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_quant_mixed_20250927_1511"
    calib_data_txt = f"{cfg.DIPOORLET.yolov8_outputs}/qnn_yolov8_calib_data_500.txt"
    onnx_path = f"{cfg.SYSTEM.MODELS_DIR}/resnet18.onnx"
    
    os.makedirs(log_dir, exist_ok=True)
    # overrides_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/inferdet/tools/yolov8_overrides_mixed.json"
    overrides_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/dipoorlet_log/3_dipoorlet_models_resnet18/snpe_dipoorlet_resnet18_mse/snpe_encodings.json"
    
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
