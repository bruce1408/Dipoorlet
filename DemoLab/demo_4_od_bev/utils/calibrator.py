import os
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


# 输入数据
input_names = [
    "img_front_fisheye",
    "img_right_fisheye",
    "img_rear_fisheye",
    "img_left_fisheye",
    "img_front_short",
    "indice_front_fisheye",
    "indice_right_fisheye",
    "indice_rear_fisheye",
    "indice_left_fisheye",
    "indice_front_short"
]



def get_calib_data_path(num_samples=256):
    img_paths = []
    data_root = "/home/bruce_ultra/data/data_sets/od_bev_0915_calib_data"
    yaml_path = os.path.join(data_root, "meta.yaml")
    
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    
    yaml_calibration_dir_list = yaml_data["calibration"]
    
    assert len(yaml_calibration_dir_list) == 256
    
    for dir_name in yaml_calibration_dir_list:
        # print(dir_name)
        dir_path = os.path.join(data_root, str(dir_name))
        for each_file_name in input_names:
            img_paths.append(os.path.join(dir_path, each_file_name + ".npz"))
            
    return img_paths
    

def Preprocess(img_path, info):
    # img = cv2.imread(img_path)
    # img_height, img_width = img.shape[:2]
    # info.update({"img_height": img_height, "img_width": img_width})
    # img = LetterBox(img, (info["input_width"], info["input_height"]))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.array(img) / 255.0
    # img = np.transpose(img, (2, 0, 1))
    # img = np.expand_dims(img, axis=0).astype(np.float32)
    
    img = np.load(img_path)
    # print(img) 
    return img[img.files[0]]
    # return img   


# For TRT
class CalibDataLoader:
    def __init__(self, batch_size, calib_count, info):
        self.data_root = "/home/bruce_ultra/data/data_sets/od_bev_0915_calib_data"
        self.info = info
        self.index = 0
        self.batch_size = batch_size
        self.calib_count = calib_count
        self.image_list = get_calib_data_path()
        self.calibration_data = np.zeros(
            (self.batch_size, 3, 720, 960), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.calib_count:
            for i in range(self.batch_size):
                image_path = self.image_list[i + self.index * self.batch_size]
                print('the image path is : ', image_path)
                image = Preprocess(image_path, self.info)
                self.calibration_data[i] = image
                if self.index >2 : break
            self.index += 1
            
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.calib_count


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
        batch = self.data_loader.next_batch()
        if not batch.size:
            return None
        cuda.memcpy_htod(self.d_input, batch)

        return [self.d_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            

if __name__ == "__main__":
    img = np.load("/home/bruce_ultra/data/data_sets/od_bev_0915_calib_data/1/img_front_fisheye.npz")
    print(img.files)
    for file_name in img.files:
        print(img[file_name])