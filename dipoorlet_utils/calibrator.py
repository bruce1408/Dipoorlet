import os,cv2
import random
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import tensorrt as trt
from tqdm import tqdm
import pycuda.driver as cuda
import pycuda.autoinit

try:
    # 尝试直接导入，适用于当前目录运行
    from common.configs import get_cfg_defaults
    config = get_cfg_defaults()
except ImportError:
    # 如果直接导入失败，尝试使用相对导入，适用于跨目录调用
    from . import get_cfg_defaults
    config = get_cfg_defaults()


class_names = config.DIPOORLET.COCO_labels

info = {
    "inputs_name": ["images"],
    "outputs_name" : ["output0"],
    "input_width": 640,
    "input_height": 640,
    "confidence_thres": 0.001,
    "iou_thres": 0.7,
    "max_det": 300,
    "class_names": class_names,
    "providers": ["CUDAExecutionProvider"]
}

current_file_path = os.path.dirname(os.path.abspath(__file__))
#200类，每类随机选5个
def get_calib_data_path():
    img_paths = []
    data_root = f"{config.DIPOORLET.imagenet_200_dir}/val/"
    data_info = pd.read_table(data_root + "val_annotations.txt")
    grouped = data_info.groupby(data_info.columns[1])
    classes = list(grouped.groups.keys())
    for cls in classes:
        group_imgs = grouped.get_group(cls).iloc[:, 0].tolist()
        random.shuffle(group_imgs)
        img_paths += group_imgs[:5]

    return img_paths

# print(get_calib_data_path())

def Preprocess(img):
    transforms_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    )
    img = transforms_val(img)
    return img

# For TRT
class CalibDataLoader:
    def __init__(self, batch_size, calib_count):
        self.data_root = f"{config.DIPOORLET.imagenet_200_dir}/val/images/"
        self.index = 0
        self.batch_size = batch_size
        self.calib_count = calib_count
        self.image_list = get_calib_data_path()
        self.calibration_data = np.zeros((self.batch_size, 3, 224, 224), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.calib_count:
            for i in range(self.batch_size):
                image_path = self.image_list[i + self.index * self.batch_size]
                image = Image.open(self.data_root + image_path).convert("RGB")
                image = Preprocess(image)
                self.calibration_data[i] = image
            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])
    
    def __len__(self):
        return self.calib_count

# 集成KL散度校准方式
class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = data_loader
        self.d_input = cuda.mem_alloc(self.data_loader.calibration_data.nbytes)
        self.cache_file = cache_file
        data_loader.reset()

    def get_batch_size(self):
        return self.data_loader.batch_size

    def get_batch(self, names):
        tensor_data = self.data_loader.next_batch()
        if not tensor_data.size:
            return None
        cuda.memcpy_htod(self.d_input, tensor_data)

        return [self.d_input]

    # 这个需要进行重写
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
    
    # 这个函数需要重写
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()

# For Dipoorlet
def get_dipoorlet_calib():
    calibration_dir_path = f"{config.DIPOORLET.dipoorlet_calib_data_dir}/mobilvnetv2_calib/input.1/"
    os.makedirs(calibration_dir_path, exist_ok=True)
        
    data_root = f"{config.DIPOORLET.imagenet_200_dir}/val/images/"
    image_list = get_calib_data_path()    
    
    for i, image_path in tqdm(enumerate(image_list)):
        image = Image.open(data_root + image_path).convert("RGB")
        image = Preprocess(image).numpy()
        
        image.tofile(f"{calibration_dir_path}" + str(i) + ".bin")
        
def LetterBox(img, new_shape):
    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img


def yolov8_process(img_path, info):
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]
    info.update({"img_height": img_height, "img_width": img_width})
    img = LetterBox(img, (info["input_width"], info["input_height"]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img 
    
    
def get_yolov8_calib(sample_num=500):
    calibration_dir_path = f"{config.DIPOORLET.dipoorlet_calib_data_dir}/yolov8_calib/images/"
    os.makedirs(calibration_dir_path, exist_ok=True)
    
    data_root = config.SYSTEM.coco2017_val_path

    image_list = os.listdir(data_root)    
    for i, image_path in tqdm(enumerate(image_list[0:sample_num])):
        image = yolov8_process(os.path.join(data_root, image_path), info)
        image.tofile(f"{calibration_dir_path}" + str(i) + ".bin")

if __name__ == "__main__":
    
    # imagenet 
    # get_dipoorlet_calib()
    
    # yolov8数据集
    get_yolov8_calib()
