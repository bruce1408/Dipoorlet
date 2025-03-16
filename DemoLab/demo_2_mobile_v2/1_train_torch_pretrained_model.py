import os
import sys
import datetime
import torch
import torch.nn as nn
from printk import * 
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import time, os, copy, numpy as np
from spectrautils import logging_utils, print_utils
from dipoorlet_utils import quant_config
from dipoorlet_utils.dataset import get_dataset
import argparse

# 添加命令行参数解析
def get_config():
    parser = argparse.ArgumentParser(description='MobileNetV2训练或继续微调')
    parser.add_argument('--resume', type=str, default=quant_config.export_work_dir, help='加载检查点文件路径继续训练')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数，如果不指定则使用配置文件中的值')
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率设置')
    parser.add_argument('--auto_resume', default=True, help='自动加载指定目录中最新的模型文件')

    args = parser.parse_args()
    return args


logger_manager = logging_utils.AsyncLoggerManager("./logs")
logger = logger_manager.logger

os.environ["CUDA_VISIBLE_DEVICES"] = quant_config.cuda_ids
num_gpus = torch.cuda.device_count()


# 添加一个函数来查找最新的模型文件
def find_latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    
    checkpoint_files = [f for f in os.listdir(directory) if f.endswith('_checkpoint.pth')]
    if not checkpoint_files:
        return None
    
    # 按照文件修改时间排序，获取最新的文件
    latest_file = max(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    return os.path.join(directory, latest_file)

def train_model(
    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, start_epoch, num_epochs=25
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    # liveloss = PlotLosses()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        print("-" * 10)
        epoch_start = time.time()
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                current_mean_loss = running_loss / ((i + 1) * inputs.size(0))  # 计算到目前为止的平均loss
                running_corrects += torch.sum(preds == labels.data)
                
                # print(
                #     "\rIteration: {}/{}, Batch Loss: {:.4f}, Average Loss: {:.4f}".format(
                #         i + 1, 
                #         len(dataloaders[phase]), 
                #         loss.item(),  # 当前batch的loss
                #         current_mean_loss  # 到目前为止的平均loss
                #     ),
                #     end="",
                # )
                # sys.stdout.flush()
                
                if i % 50 == 0:
                    logger.info(
                        "Epoch:[{}|{}], Iteration: {}/{}, Loss: {:.4f}, Average Loss: {:.4f}".format(
                            epoch + 1,
                            num_epochs, 
                            i + 1, 
                            len(dataloaders[phase]), 
                            loss.item(),  # 当前batch的loss
                            current_mean_loss  # 到目前为止的平均loss
                        )
                    )

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "train":
                avg_loss = epoch_loss
                t_acc = epoch_acc
                scheduler.step()
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc

            if phase == "val" and epoch_acc > best_acc:
                logger.info("Validation Accuracy after epoch {}: {:.4f}".format(epoch + 1, epoch_acc))
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                os.makedirs(quant_config.export_work_dir, exist_ok=True)
                # torch.save(model, f"{quant_config.export_work_dir}/best_model.pth")
                
                # 保存每个epoch的模型权重
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                }, f"{quant_config.export_work_dir}/mobile_v2_epoch_{epoch + 1}_checkpoint.pth")


        # 用列表存储所有的输出信息
        epoch_time = time.time() - epoch_start
        print()
        epoch_summary = [
            "Epoch Summary:",
            "  Train Loss: {:.4f} | Train Accuracy: {:.4f}".format(avg_loss, t_acc),
            "  Val Loss: {:.4f} | Val Accuracy: {:.4f}".format(val_loss, val_acc),
            "  Best Val Accuracy So Far: {:.4f}".format(best_acc),
            "  Epoch Time: {:.0f}m {:.0f}s".format(epoch_time // 60, epoch_time % 60)
        ]
        
        print_colored_box(epoch_summary, text_color='green', box_color='yellow')

    time_elapsed = time.time() - since
    print_colored_box(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    
    print_colored_box("Best val Acc: {:4f}".format(best_acc), attrs=['bold'], text_color='green', box_color='yellow')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# model = models.resnet18(pretrained=True)
model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

# Finetune Final few layers to adjust for tiny imagenet input
# model.avgpool = nn.AdaptiveAvgPool2d(1)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 200)

model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 200)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if num_gpus > 1:
    # 使用所有可见的 GPU
    device_ids = list(range(num_gpus))  # 此时 num_gpus 会是 1，因为我们只暴露了一个 GPU
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    logger.info(f"Using {num_gpus} GPUs for training: {device_ids}")
else:
    logger.info("Only one GPU or no GPU available, using single GPU or CPU.")

model = model.to(device)

args = get_config()

# Multi GPU
# model = torch.nn.DataParallel(model, device_ids=[0, 7])

# Loss Function
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

train_dataset, val_dataset, _ = get_dataset()

train_loaders = torch.utils.data.DataLoader(
    train_dataset, batch_size=quant_config.train_batch_size, shuffle=True, num_workers=8
)
val_loaders = torch.utils.data.DataLoader(
    val_dataset, batch_size=quant_config.val_batch_size, shuffle=True, num_workers=8
)


dataloaders = {}
dataloaders["train"] = train_loaders
dataloaders["val"] = val_loaders

dataset_sizes = {}
dataset_sizes["train"] = len(train_dataset)
dataset_sizes["val"] = len(val_dataset)



# 如果指定了 resume 参数，加载检查点
start_epoch = 0

# 自动查找最新的检查点文件
if args.auto_resume and os.path.isdir(args.resume):
    latest_checkpoint = find_latest_checkpoint(args.resume)
    if latest_checkpoint:
        args.resume = latest_checkpoint
        logger.info(f"自动加载最新的检查点文件: {latest_checkpoint}")
    else:
        logger.info(f"在目录 {args.resume} 中未找到检查点文件，将从头开始训练。")
        args.resume = None

if args.resume is not None and os.path.isfile(args.resume):
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
    exp_lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    logger.info("Resuming training from epoch {} with best_acc: {:.4f}".format(start_epoch, checkpoint.get('best_acc', 0.0)))
else:
    logger.info("Starting training from scratch.")

# 如果命令行传入了 epochs 参数，则覆盖配置文件中的 epoch 数值
num_epochs = args.epochs if args.epochs is not None else quant_config.epochs

model = train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    start_epoch,
    num_epochs=quant_config.epochs,
)

current_timestamp = datetime.datetime.now()
formatted_timestamp = current_timestamp.strftime("%Y_%m_%d")
torch.save(model, f"{quant_config.export_work_dir}/{formatted_timestamp}_mobilev2_model.pth")