import os, sys
import subprocess
import datetime
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

def main():
    # 生成时间戳，格式：YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    param_is_symmetric = True
    act_is_symmetric = True
    is_per_channel = False
    is_per_row = False
    
    # 命名规则按照 = 平台+模型+量化工具+量化算法+时间戳
    log_dir = (f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_int8_100_{timestamp}_version_0241_"
               f"param_{param_is_symmetric}_act_{act_is_symmetric}_perchannel_{is_per_channel}")

    
    # log_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_int8_120"


    os.makedirs(log_dir, exist_ok=True)
    
    print(f"输出目录已创建: {log_dir}")
    print(f"时间戳: {timestamp}")

    calib_data_txt = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_calib_data_100.txt"
    onnx_path = f"{cfg.SYSTEM.MODELS_DIR}/resnet18.onnx"

    
    # 构建 qnn-onnx-converter 命令
    command = [
        "qnn-onnx-converter",
        "--input_network", onnx_path,
        "--input_list", calib_data_txt,
        "-o", f"{log_dir}/qnn_resnet18_quant_basic.cpp",
    ]
    
    # 根据 param_is_symmetric 添加参数
    if param_is_symmetric:
        command.extend(["--param_quantizer_schema", "symmetric"])
        print(f"参数设置: 权重使用对称量化 (symmetric)")
    else:
        command.extend(["--param_quantizer_schema", "asymmetric"])
        print(f"参数设置: 权重使用非对称量化 (asymmetric)")
        
    if act_is_symmetric:
        command.extend(["--act_quantizer_schema", "symmetric"])
        print(f"参数设置: 激活使用对称量化 (symmetric)")
    else:
        command.extend(["--act_quantizer_schema", "asymmetric"])
        print(f"参数设置: 激活使用非对称量化 (asymmetric)")
        
    if is_per_channel:
        command.extend(["--use_per_channel_quantization"])
        print(f"参数设置: 权重使用逐通道量化")
        
    if is_per_row:
        command.extend(["--use_per_row_quantization"])
        print(f"参数设置: 权重使用逐行量化")
    # else:
    #     command.extend(["--use_per_channel_quantization"])
    #     print(f"参数设置: 权重使用逐通道量化")
    
    print(f"执行命令: {' '.join(command)}")
    
    # 执行命令
    subprocess.run(command, check=True)
    
    print(f"转换完成！输出文件保存在: {log_dir}")

    
if __name__ == "__main__":
    main()
