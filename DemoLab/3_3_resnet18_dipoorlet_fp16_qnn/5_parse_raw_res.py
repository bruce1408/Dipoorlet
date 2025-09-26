import cv2
from pathlib import Path
import os, sys
import torch
import subprocess
import numpy as np
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
from torchvision import transforms
import dipoorlet_utils.quant_config as config
from common.configs import get_cfg_defaults
from spectrautils.print_utils import *
cfg = get_cfg_defaults()

def parse_labels_from_file(file_path):
    """
    这个函数负责读取并解析标签文件。

    它会打开你指定的 txt 文件，一行一行地读取内容，
    然后把每一行的数字和英文标签提取出来，
    最后将它们存到一个字典里。

    参数:
    file_path (str): 标签文件的路径。

    返回:
    dict: 一个包含所有标签的字典，键是数字，值是对应的英文。
          如果文件找不到，会返回一个空字典。
    """
    labels_map = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        index = int(parts[0])
                        
                        label_part = parts[1].strip()
                        if label_part.endswith(','):
                            label_part = label_part[:-1]
                        label = label_part.strip("'")
                        labels_map[index] = label

    except FileNotFoundError:
        # 如果找不到文件，就打印一个错误提示。
        print(f"错误：找不到文件 '{file_path}'。请检查文件名和路径是否正确。")
    
    return labels_map

def preprocess_image(image_path: str) -> np.ndarray:
    """
    对输入的单张图片进行预处理，使其符合 ResNet-18 的输入要求。
    此版本使用了 torchvision.transforms，与 PyTorch 训练流程保持一致。

    Args:
        image_path (str): 输入图片的路径。

    Returns:
        np.ndarray: 经过预处理后的图像数据，格式为 NCHW，数据类型为 float32。
    """
    # print("使用 torchvision.transforms 进行图像预处理...")
    
    # 1. 打开图片
    img_pil = Image.open(image_path).convert('RGB')

    # 2. 定义验证集的图像变换逻辑 (根据你提供的代码)
    # 这些是 ImageNet 数据集上常用的标准化参数
    IMG_SIZE = 224
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize(248),       # 先将图片短边缩放到248
        transforms.CenterCrop(IMG_SIZE), # 再从中心裁剪出224x224
        transforms.ToTensor(),        # 转换成Tensor，并将像素值缩放到[0,1]
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD) # 标准化
    ])

    # 3. 应用变换
    # transform 函数会返回一个形状为 (C, H, W) 的 PyTorch Tensor
    img_tensor = transform(img_pil)

    # 4. 添加 Batch 维度: CHW -> NCHW (1, C, H, W)
    #    ONNX 模型需要一个4维的输入 (N代表批次数)
    img_tensor = img_tensor.unsqueeze(0)

    # 5. 将 PyTorch Tensor 转换为 NumPy 数组
    #    onnxruntime 的输入需要是 NumPy 数组
    input_numpy = img_tensor.numpy()

    # print(f"预处理后图像的形状: {input_numpy.shape}")
    return input_numpy

def infer_with_onnx(model_path: str, image_path: str, labels_map: dict, single_pic: bool = False):
    """
    使用 ONNX Runtime 对单张图片进行 ResNet-18 推理。
    (这个函数内部没有任何改动)
    Args:
        model_path (str): .onnx 模型的路径。
        image_path (str): 输入图片的路径。
    """
    
    session = ort.InferenceSession(model_path)
    
    input_name = session.get_inputs()[0].name

    input_tensor = preprocess_image(image_path)

    result = session.run(None, {input_name: input_tensor})
    output_tensor = result[0]
    

    predicted_class_id = np.argmax(output_tensor)
    confidence_score = np.max(output_tensor)

    if single_pic:
        print("\n" + "="*24 + " onnx 推理结果 " + "="*23)
        print(f"预测的类别ID: {predicted_class_id}")
        print(image_path)
        print(f"预测的类别是: {labels_map.get(predicted_class_id)}")
        print("="*62)
    return predicted_class_id, confidence_score


def parse_raw_data(raw_file_path):
    raw_data = np.fromfile(raw_file_path, dtype=np.float32)
    raw_data = raw_data.reshape((1, 1000))
    predicted_class_id = np.argmax(raw_data)

    return predicted_class_id, raw_data

             
def main(info, mode):
    img_path = f"{cfg.SYSTEM.imagenet_dir}/val_mini/n02687172/ILSVRC2012_val_00046511.JPEG"
    
    _, info = preprocess(img_path, info)
    
    raw_file_path = f"{log_dir}/Result_0/output0.raw"
    raw_data = np.fromfile(raw_file_path, dtype=np.float32)
    raw_data = raw_data.reshape(info["output_shape"])
    
    results, info = postprocess(raw_data, info)
    
    show_results_single_img(img_path, results, class_names, f"{log_dir}/test_res_qnn_{mode}_556000.jpg")
    print_colored_text(f"pic saved in :\n{log_dir}/test_res_qnn_{mode}_556000.jpg", "green")


if __name__ == "__main__":
    
    ONNX_MODEL_PATH = f"{cfg.SYSTEM.MODELS_DIR}/resnet18.onnx"
    LABEL_PATH = "/mnt/share_disk/bruce_trie/workspace/imagenet1000_clsidx_to_labels.txt"
    INFERENCE_SINGLE_PIC = False
    COMPARE_WITH_ONNX = False
    labels_map = parse_labels_from_file(LABEL_PATH)


    if INFERENCE_SINGLE_PIC:
        IMAGE_PATH = f"{cfg.SYSTEM.imagenet_dir}/val_mini/n02687172/ILSVRC2012_val_00048573.JPEG"
        IMAGE_PATH = f"{cfg.SYSTEM.imagenet_dir}/val_mini/n03527444/ILSVRC2012_val_00046409.JPEG"
        infer_with_onnx(ONNX_MODEL_PATH, IMAGE_PATH, labels_map, INFERENCE_SINGLE_PIC)
        RAW_FILE_PATH = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_int8_100_20250924_113605_debug/Result_1/_191.raw"
        RAW_FILE_PATH = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_quant_fp16_20250925_1930/qnn_resnet18_quant_fp16_infer_res/Result_0/_191.raw"
        RAW_FILE_PATH = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_quant_fp16_output/Result_3/_191.raw"
        predicted_class_id, _ = parse_raw_data(RAW_FILE_PATH)
        print(f"qnn 预测的类别ID: {predicted_class_id}")
    
    else:
        total_num = 0
        correct_num = 0
        raw_file_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_quant_fp16_20250925_1930/qnn_resnet18_quant_fp16_infer_res"
        raw_file_dir = f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_quant_fp16_output_1"
        raw_file_path = Path(raw_file_dir)
        raw_dir_lists = [d.name for d in raw_file_path.iterdir() if d.is_dir()]
                
        with open(f"{cfg.DIPOORLET.resnet18_outputs}/qnn_resnet18_jpg_data.txt", "r") as f:
            lines = f.readlines()
        
        # 对qnn推理结果进行排序，保证推理结果的顺序与原始图片的顺序一致
        sorted_raw_dir_lists = sorted(raw_dir_lists, key=lambda x: int(x.split("_")[-1]))
        progress_bar = tqdm(zip(sorted_raw_dir_lists, lines), total=len(lines), desc="正在比较推理结果")
        
        if COMPARE_WITH_ONNX:
            for each_dir, jpg_path in progress_bar:
                raw_dir_path = os.path.join(raw_file_dir, each_dir, "_191.raw")
                predicted_class_id, _ = parse_raw_data(raw_dir_path)
                infer_class_id, _ = infer_with_onnx(ONNX_MODEL_PATH, jpg_path.strip(), labels_map)
                if predicted_class_id == infer_class_id:
                    correct_num += 1
                total_num += 1
            print(f"total_num: {total_num}, correct_num: {correct_num}, accuracy: {correct_num / total_num}")
        else:
            with open("/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/Dipoorlet/DemoLab/3_3_resnet18_dipoorlet_fp16_qnn/class_to_idx.txt", "r") as f:
                class_to_idx = f.readlines()
                class_to_labels = {int(line.strip().split(":")[1]): line.strip().split(":")[0] for line in class_to_idx}
            
            for each_dir, jpg_path in progress_bar:
                dir_label = jpg_path.strip().split("/")[-2]
                raw_dir_path = os.path.join(raw_file_dir, each_dir, "_191.raw")
                predicted_class_id, _ = parse_raw_data(raw_dir_path)
                if dir_label == class_to_labels[predicted_class_id]:
                    correct_num += 1
                total_num += 1
            print(f"total_num: {total_num}, correct_num: {correct_num}, accuracy: {correct_num / total_num}")                