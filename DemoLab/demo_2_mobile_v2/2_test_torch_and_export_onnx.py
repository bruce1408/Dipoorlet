import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt
import time, os, copy, numpy as np
from tqdm import tqdm
import config
from dataset import get_dataset
from printk import *

current_file_path = os.path.dirname(os.path.abspath(__file__))

# model = torch.load(f"{current_file_path}/models/2024_10_30_mobilev2_model.pth")
model = torch.load(f"{config.train_mobile_v2_dir}/2024_10_30_mobilev2_model.pth")
_, val_dataset, _ = get_dataset()
dataloaders = torch.utils.data.DataLoader(
    val_dataset, batch_size=128, shuffle=True, num_workers=8
)
running_corrects = 0.0
for i, (inputs, labels) in tqdm(enumerate(dataloaders)):
    inputs = inputs.cuda()
    labels = labels.cuda()
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds == labels.data)
print_colored_box(f"Accuracy : {running_corrects / len(val_dataset) * 100:.2f}%")

# convert to onnx
if isinstance(model, torch.nn.DataParallel):
    model = model.module


x = torch.randn(1, 3, 224, 224).cuda()

export_onnx_path = f"{config.train_mobile_v2_dir}/mobilev2_model_new.onnx"
torch.onnx.export(
    model, 
    x, 
    export_onnx_path, 
    export_params=True, 
    opset_version=11
)
print_colored_box(f"onnx has been saved in {export_onnx_path}")
