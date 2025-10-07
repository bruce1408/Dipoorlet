import sys,os
from tqdm import tqdm
import torch, torchvision
import torch.nn as nn
import progressbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt
import time, copy, numpy as np
from common.configs import get_cfg_defaults
from dipoorlet_utils.dataset import get_dataloaders
from spectrautils import print_utils
from dipoorlet_utils.dataset import ImageNetEvaluator

cfg = get_cfg_defaults()



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
    
    
def evaluate_model():
    
    # normal or tiny
    imagenet_mode = "normal"
    
    current_file_path = os.path.dirname(os.path.abspath(__file__))

    # load resnet18 model
    model = model.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)   
    # model = torch.load("/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/aimet_log/resnet18_cle_bc_pc/resnet_model_cle_bc.pt")
    
    # load mobile_v2 model
    # model = torch.load("/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/models/mobile_v2_best_model_200_labels.pth")
    
    model.cuda()

    model.eval()
    
    if imagenet_mode == "normal":
        datasets_dir = cfg.SYSTEM.imagenet_dir
    else:
        datasets_dir = cfg.SYSTEM.imagenet_200_dir
    
    _, val_dataset, _ = get_dataloaders(
        datasets_dir=datasets_dir,
        imagenet_mode=imagenet_mode,
        batch_size=cfg.DIPOORLET.val_batch_size
    )

    
    sample_nums = len(val_dataset.dataset)    
    evaluator = ImageNetEvaluator(
        datasets_dir, 
        image_size=224,
        batch_size=40,
        num_workers=16
    )
    evaluator.evaluate(model, use_cuda=True)
    
    
    # ==================== 简易计算版本 =========================
    # running_corrects = 0.0
    # for i, (inputs, labels) in tqdm(enumerate(val_dataset)):
    #     inputs = inputs.cuda()
    #     labels = labels.cuda()
    #     outputs = model(inputs)
    #     _, preds = torch.max(outputs, 1)
    #     running_corrects += torch.sum(preds == labels.data)
    # print_utils.print_colored_box(f"Accuracy : {running_corrects / sample_nums * 100:.2f}%")
    # # resnet18 在val_mini数据集上的准确率是 70.85%
    # ==================== 简易计算版本 =========================
    
    return model


def export_onnx(model):
    
    # convert to onnx
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    x = torch.randn(1, 3, 224, 224).cuda()

    export_onnx_path = f"{cfg.SYSTEM.MODELS_DIR}/resnet18_cle_aimet.onnx"
    torch.onnx.export(
        model, 
        x, 
        export_onnx_path, 
        export_params=True, 
        opset_version=11
    )

    print_utils.print_colored_text(f"onnx has been saved in {export_onnx_path}")


if __name__ == "__main__":
    model = evaluate_model()
    
    export_onnx(model)