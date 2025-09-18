import sys,os
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
import time, copy, numpy as np
from tqdm import tqdm
from common.configs import get_cfg_defaults
from dipoorlet_utils.dataset import get_dataset
from spectrautils import print_utils


cfg = get_cfg_defaults()
current_file_path = os.path.dirname(os.path.abspath(__file__))

model = torch.load(f"{cfg.DIPOORLET.export_work_dir}/best_model.pth")

_, val_dataset, _ = get_dataset(cfg.DIPOORLET.imagenet_200_dir)

dataloaders = torch.utils.data.DataLoader(
    val_dataset, batch_size=cfg.DIPOORLET.val_batch_size, shuffle=True, num_workers=8
)

running_corrects = 0.0
for i, (inputs, labels) in tqdm(enumerate(dataloaders)):
    inputs = inputs.cuda()
    labels = labels.cuda()
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds == labels.data)
print_utils.print_colored_box(f"Accuracy : {running_corrects / len(val_dataset) * 100:.2f}%")

# convert to onnx
if isinstance(model, torch.nn.DataParallel):
    model = model.module


x = torch.randn(1, 3, 224, 224).cuda()

export_onnx_path = f"{cfg.DIPOORLET.export_work_dir}/mobilev2_model_new.onnx"
torch.onnx.export(
    model, 
    x, 
    export_onnx_path, 
    export_params=True, 
    opset_version=11
)
print_utils.print_colored_box(f"onnx has been saved in {export_onnx_path}")
