import os
from qnnhelper.helper import MultiStageHelper
from qnnhelper.utils import symlink_previous_outputs_with_dataset

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)

cfg = "./config/config.yaml"
model = f"/share/zhangshuo/model_v5/modelv5.onnx"
work_dir = f"/share/yangcun/370_pod/"

exec_num = 100

helpers = MultiStageHelper(cfg=cfg, model=model, work_dir=work_dir, validate_mode=True)

helpers.evaluate(idx=0, exec_num=exec_num)

symlink_previous_outputs_with_dataset(
    os.path.join(work_dir, "0", "results"), os.path.join(work_dir, "mid", "results")
)

helpers.evaluate(idx=1, exec_num=exec_num)


# helpers.speedtest()
# helpers.direct_onnx_infer(num=exec_num)
