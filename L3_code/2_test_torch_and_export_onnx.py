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
from dataset import get_dataset
from printk import *


model = torch.load("/mnt/share_disk/bruce_trie/Quantizer-Tools/Dipoorlet/L3_code/models/2024_10_30_mobilev2_model.pth")

_, val_dataset, _ = get_dataset()
dataloaders = torch.utils.data.DataLoader(
    val_dataset, batch_size=128, shuffle=True, num_workers=8
)
running_corrects = 0.0
for i, (inputs, labels) in enumerate(dataloaders):
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

torch.onnx.export(
    model, 
    x, 
    "/mnt/share_disk/bruce_trie/Quantizer-Tools/Dipoorlet/L3_code/models/mobilev2_model_new.onnx", 
    export_params=True, 
    opset_version=11
)