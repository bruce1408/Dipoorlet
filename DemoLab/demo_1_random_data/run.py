import os
import subprocess
import config
from printk import print_colored_box 

# 获取当前脚本所在的目录
workdir = os.path.dirname(os.path.abspath(__file__))

# 切换到该目录
os.chdir(workdir)

# 检查并创建 log_dir 目录
log_dir = os.path.join(workdir, 'log_dir')
os.makedirs(log_dir, exist_ok=True)
onnx_path = os.path.join(config.model_dir, 'resnet34_model.onnx')
calibration_path = config.calibration_dir

# 定义 torchrun 命令及其参数
command = [
    'torchrun',
    '-m', 'dipoorlet',
    '-M', f'{onnx_path}',
    '-I', f'{calibration_path}',
    '-N', '2',
    '-A', 'minmax',
    '-D', 'trt',
    '-O', f'{log_dir}',
    '--adaround',
    '--ada_epoch', '1'
]

# 运行命令
try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print_colored_box(f"Error occurred while running torchrun: {e}")
