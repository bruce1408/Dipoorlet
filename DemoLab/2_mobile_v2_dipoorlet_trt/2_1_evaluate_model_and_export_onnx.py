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
from dipoorlet_utils.dataset import get_dataloaders
from spectrautils import print_utils


cfg = get_cfg_defaults()

def evaluate_model(imagenet_mode):
    
    current_file_path = os.path.dirname(os.path.abspath(__file__))

    model = torch.load(f"{cfg.SYSTEM.MODELS_DIR}/mobile_v2_best_model_basic_tiny.pth")
    # model = torch.load("/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/models/2025_09_18_mobilev2_model.pth")
    # model_torch.load("/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/models/2025_09_18_mobilev2_model.pth")
    
    if imagenet_mode == "normal":
        datasets_dir = cfg.SYSTEM.imagenet_dir
    else:
        datasets_dir = cfg.SYSTEM.imagenet_200_dir

    _, val_dataset, _ = get_dataloaders(
        datasets_dir=datasets_dir,
        imagenet_mode=imagenet_mode,
        batch_size=cfg.DIPOORLET.val_batch_size
    )

    # dataloaders = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=cfg.DIPOORLET.val_batch_size, shuffle=True, num_workers=8)

    total_samples = len(val_dataset.dataset)
    running_corrects = 0.0
    for i, (inputs, labels) in tqdm(enumerate(val_dataset)):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    print_utils.print_colored_box(f"Accuracy : {running_corrects / total_samples * 100:.2f}%")


def export_onnx():
    
    # convert to onnx
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    x = torch.randn(1, 3, 224, 224).cuda()

    export_onnx_path = f"{cfg.SYSTEM.MODELS_DIR}/mobilev2_model_trained.onnx"
    torch.onnx.export(
        model, 
        x, 
        export_onnx_path, 
        export_params=True, 
        opset_version=11
    )

    print_utils.print_colored_text(f"onnx has been saved in {export_onnx_path}")


if __name__ == "__main__":
    imagenet_mode = "tiny"

    evaluate_model(imagenet_mode)
    
    # export_onnx()