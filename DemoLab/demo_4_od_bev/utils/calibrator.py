import os,sys
import random
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import yaml
import json
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import demo_utils.quant_config as config


# 输入数据
input_names = [
    "front_short_camera", 
    "front_fisheye_camera",
    "right_fisheye_camera",
    "rear_fisheye_camera", 
    "left_fisheye_camera", 
    "indices"
]



def get_calib_data_path(num_samples=256, yaml=None):
    img_paths = []
    dir_name_list = []
    data_root = f"{config.od_bev_calib_dir}/od_bev_1125_calib_data_v3_126"
    if yaml != None:
        yaml_path = os.path.join(data_root, "meta.yaml")
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        dir_name_list = yaml_data["calibration"]
        assert len(dir_name_list) == num_samples
    
    else:
        dir_name_list = os.listdir(data_root)
        
    for dir_name in dir_name_list:
        image_dir_list = []
        dir_path = os.path.join(data_root, str(dir_name))
        for each_file_name in input_names:
            image_dir_list.append(os.path.join(dir_path, each_file_name + ".raw"))
        img_paths.append(image_dir_list)
    return img_paths
    

def Preprocess(img_path, idx, info, raw_format=True):
    # img = cv2.imread(img_path)
    # img_height, img_width = img.shape[:2]
    # info.update({"img_height": img_height, "img_width": img_width})
    # img = LetterBox(img, (info["input_width"], info["input_height"]))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.array(img) / 255.0
    # img = np.transpose(img, (2, 0, 1))
    # img = np.expand_dims(img, axis=0).astype(np.float32)
    
    # img_data = []
    # for i in range(len(info["input_width"])):
    array_from_raw = np.fromfile(img_path, dtype=np.float32)
    # print("======= the image_path is ", img_path)
    # 重新调整数组的形状
    if idx != len(info["inputs_name"]) - 1:
        if raw_format:
            array_from_raw = array_from_raw.reshape((1, 3, info["input_width"][idx], info["input_height"][idx]))
        else:
            array_from_raw = np.load(img_path)
            array_from_raw = array_from_raw[array_from_raw.files[0]]
    else:
        # 处理 indices tensor
        if raw_format:
            # print("reshape indices")
            array_from_raw = array_from_raw.reshape(info["indices"])
        else:
            array_from_raw = np.load(img_path)
            array_from_raw = array_from_raw[array_from_raw.files[0]]
            
    # img_data.append(array_from_raw)
    return array_from_raw


# For TRT
class CalibDataLoader:
    def __init__(self, batch_size, calib_count, info):
        self.data_root = f"{config.od_bev_calib_dir}/od_bev_1125_calib_data_v3_126"
        self.info = info
        self.index = 0
        self.batch_size = batch_size
        print(self.batch_size)
        self.calib_count = calib_count
        self.image_list = get_calib_data_path()
        self.calibration_data = [
            np.zeros((self.batch_size, 3, 720, 1920), dtype=np.float32),
            np.zeros((self.batch_size, 3, 720, 960), dtype=np.float32),
            np.zeros((self.batch_size, 3, 720, 960), dtype=np.float32),
            np.zeros((self.batch_size, 3, 720, 960), dtype=np.float32),
            np.zeros((self.batch_size, 3, 720, 960), dtype=np.float32),
            np.zeros((5, 256, 192, 4, 2), dtype=np.float32)
        ]
        # self.calibration_data = None, None, None, None, None, None 
        

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.calib_count:
            for i in range(self.batch_size):
                image_path = self.image_list[self.index]
                print('======== the image path is : ', image_path)
                image_data = Preprocess(image_path[i], i, self.info)
                self.calibration_data[i] = image_data
                # if self.index > 2 : break
            self.index += 1
            
            self.calibration_data = [np.ascontiguousarray(data, dtype=np.float32) for data in self.calibration_data]

            
            # return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
            return self.calibration_data            
        else:
            return np.array([])

    def __len__(self):
        return self.calib_count


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = data_loader
        self.d_input = []  # 初始化 d_input 列表

        for input_data in self.data_loader.calibration_data:
            self.d_input.append(cuda.mem_alloc(input_data.nbytes))
        # self.d_input = cuda.mem_alloc(self.data_loader.calibration_data.nbytes)
        self.cache_file = cache_file
        data_loader.reset()

    def get_batch_size(self):
        return self.data_loader.batch_size

    def get_batch(self, names):
        batch = self.data_loader.next_batch()
        if not batch:
            return None
        # if not batch.size:
            # return None
        # 将每个输入数据逐个传送到对应的设备内存中
        for i, input_data in enumerate(batch):
            cuda.memcpy_htod(self.d_input[i], input_data)


        # cuda.memcpy_htod(self.d_input, batch)

        return self.d_input

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            

if __name__ == "__main__":
    # img = np.load("/home/bruce_ultra/data/data_sets/od_bev_0915_calib_data/1/img_front_fisheye.npz")
    # print(img.files)
    # for file_name in img.files:
    #     print(img[file_name])
    
    
    info = {
        "inputs_name": [
            "front_short_camera", 
            "front_fisheye_camera",
            "right_fisheye_camera",
            "rear_fisheye_camera", 
            "left_fisheye_camera", 
            "indices"
        ],
        "outputs_name" : [
            "dim", 
            "height", 
            "reg",
            "rot",
            "hm"
        ],
        "input_width": [1920, 960, 960, 960, 960],
        "input_height": [720, 720, 720, 720, 720],
        "indices": [5, 256, 192, 4, 2],
        "confidence_thres": 0.001,
        "iou_thres": 0.7,
        "max_det": 300,
        "providers": ["CUDAExecutionProvider"]
    }
    
    
    img_path_list = get_calib_data_path()
    print(len(img_path_list))
    dataloader = CalibDataLoader(batch_size=1, calib_count=128, info=info)

    