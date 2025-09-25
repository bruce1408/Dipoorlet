import os
import datetime
import sys

from qnnhelper.helper import MultiStageHelper
from qnnhelper.utils import symlink_previous_outputs_with_dataset

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)

details = ''
# 获取命令行参数的个数
argc = len(sys.argv)
if argc > 1:
    details = sys.argv[1]

# 获取当前时
current_time = datetime.datetime.now()

# 格式化时间为 day-hour-minute
formatted_time = current_time.strftime("%Y%m%d_%H%M%S")

calib_num = 256
exec_num = 256


cfg = "./config/config_only_one_stage.yaml"
model = f"/mnt/share_disk/bruce_trie/onnx_models/modelv5_0915.onnx"
work_dir = f"/mnt/share_disk/bruce_trie/outputs/od_bev_0915_int8_{formatted_time}_{calib_num}"


helpers = MultiStageHelper(cfg=cfg, model=model, work_dir=work_dir)

helpers.compile(idx=0, calib_num=calib_num, exec_num=exec_num, execute=True, remote=False)


