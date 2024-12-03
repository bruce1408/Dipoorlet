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
formatted_time = current_time.strftime("%m%d_%H%M")

cfg = "./config/config.yaml"
model = f"/share/zhangshuo/model_v5/modelv5.onnx"

# file_path = f"/share/yangcun/370_pod/v5"
work_dir = f"/mnt/share_disk/bruce_trie/report_outputs/v5_od_bev_outputs/two_stage"
# work_dir = file_path_with_time = f"{file_path}_{formatted_time}_{details}"


calib_num = 100
exec_num = 100

helpers = MultiStageHelper(cfg=cfg, model=model, work_dir=work_dir)

helpers.compile(idx=0, exec_num=exec_num, calib_num=calib_num, dump_override=True)

symlink_previous_outputs_with_dataset(
    os.path.join(work_dir, "0", "results"), os.path.join(work_dir, "mid", "results")
)

exec_num_1 = 100


helpers.compile(idx=1, exec_num=exec_num_1, calib_num=calib_num, override=True,remote=False)
