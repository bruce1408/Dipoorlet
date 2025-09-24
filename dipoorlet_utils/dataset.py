import os
import sys
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import logging
import progressbar
from typing import NoReturn
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from loguru import logger
from Examples.common.utils import accuracy
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader

logger = logging.getLogger('Eval')

# =============================================================================
# 1. 全局常量和配置
# =============================================================================
IMG_SIZE = (224, 224)
# ImageNet 数据集常用的统计值
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
logger = logging.getLogger(__name__)


def scan_and_write_paths(directory: str, output_file: str, sample_num: int = 100) -> None:
    """
    扫描指定目录下的所有文件，并将它们的完整路径写入一个文本文件中，每个路径占一行。

    这个函数会递归地遍历所有子目录。

    Args:
        directory (str): 你想要扫描的目录的路径。
        output_file (str): 用来保存结果的 .txt 文件的路径。
    """
    paths_written = 0
    # 使用 'try...except' 结构来捕捉可能发生的错误，比如目录不存在。
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # os.walk() 是一个非常有用的函数，它可以遍历一个目录树。
            # root: 当前正在遍历的文件夹路径。
            # dirs: 当前文件夹中的子文件夹列表。
            # files: 当前文件夹中的文件列表。
            for root, dirs, files in os.walk(directory):
                # 遍历当前文件夹下的所有文件名。
                for file in files:
                    if sample_num is not None and paths_written >= sample_num:
                        return 

                    # 使用 os.path.join() 来创建一个完整的、跨平台兼容的文件路径。
                    full_path = os.path.join(root, file)
                    
                    # 将完整路径写入文件，并在末尾添加一个换行符 '\n'。
                    f.write(full_path + '\n')
                    
                    paths_written += 1
        
        print(f"成功！所有文件路径已保存到 {output_file}")

    except FileNotFoundError:
        print(f"错误：目录 '{directory}' 不存在，请检查路径是否正确。")
    except Exception as e:
        print(f"发生了一个预料之外的错误: {e}")

    

class ImageNetEvaluator:
    """
    For validation of a trained model using the ImageNet dataset.
    """

    def __init__(self, images_dir: str, image_size: int, batch_size: int = 128,
                 num_workers: int = 32, num_val_samples_per_class: int = None):
        """
        :param images_dir: The path to the data directory
        :param image_size: The length of the image
        :param batch_size: The batch size to use for training and validation
        :param num_workers: Indiicates to the data loader how many sub-processes to use for data loading.
        :param num_train_samples_per_class: Number of samples to use per class.
        """
        
        
        self._val_data_loader = ImageNetDataLoader(images_dir,
                                                   image_size=image_size,
                                                   batch_size=batch_size,
                                                   is_training=False,
                                                   num_workers=num_workers,
                                                   num_samples_per_class=num_val_samples_per_class).data_loader

    def evaluate(self, model: nn.Module, iterations: int = None, use_cuda: bool = False) -> float:
        """
        Evaluate the specified model using the specified number of samples batches from the
        validation set.
        :param model: The model to be evaluated.
        :param iterations: The number of batches to use from the validation set.
        :param use_cuda: If True then use a GPU for inference.
        :return: The accuracy for the sample with the maximum accuracy.
        """

        device = torch.device('cpu')
        if use_cuda:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                logger.error('use_cuda is selected but no cuda device found.')
                raise RuntimeError("Found no CUDA Device while use_cuda is selected")

        if iterations is None:
            logger.info('No value of iteration is provided, running evaluation on complete dataset.')
            iterations = len(self._val_data_loader)
        if iterations <= 0:
            logger.error('Cannot evaluate on %d iterations', iterations)

        acc_top1 = 0
        acc_top5 = 0

        logger.info("Evaluating nn.Module for %d iterations with batch_size %d",
                    iterations, self._val_data_loader.batch_size)

        model = model.to(device)
        model = model.eval()
        
        batch_cntr = 1
        with progressbar.ProgressBar(max_value=iterations) as progress_bar:
            with torch.no_grad():
                for input_data, input_label in self._val_data_loader:

                    inputs_batch = input_data.to(device)
                    target_batch = input_label.to(device)

                    predicted_batch = model(inputs_batch)

                    batch_avg_top_1_5 = accuracy(output=predicted_batch, target=target_batch,
                                                 topk=(1, 5))

                    acc_top1 += batch_avg_top_1_5[0].item()
                    acc_top5 += batch_avg_top_1_5[1].item()

                    progress_bar.update(batch_cntr)

                    batch_cntr += 1
                    if batch_cntr > iterations:
                        break

        acc_top1 /= iterations
        acc_top5 /= iterations

        logger.info('Avg accuracy Top 1: %f Avg accuracy Top 5: %f on validation Dataset', acc_top1, acc_top5)
        
        return acc_top1




# =============================================================================
# 2. 为 "tiny" 模式定制的 Dataset 类 (来自您的第一个脚本)
#    用于处理 Tiny ImageNet 的特殊文件结构
# =============================================================================
class TinyImageNetDataset(Dataset):
    def __init__(self, files, labels, encoder, transforms, mode):
        super().__init__()
        self.files = files
        self.labels = labels
        self.encoder = encoder
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        pic = Image.open(self.files[index]).convert("RGB")

        if self.mode in ["train", "val"]:
            x = self.transforms(pic)
            label = self.labels[index]
            y = self.encoder.transform([label])[0]
            return x, y
        elif self.mode == "test":
            x = self.transforms(pic)
            return x, self.files[index]


# =============================================================================
# 3. 为 "normal" 模式定制的 Dataset 类 (来自 AIMET 脚本)
#    功能类似于 torchvision.datasets.ImageFolder，更通用
# =============================================================================
class StandardImageFolder(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.loader = default_loader
        
        classes, class_to_idx = self._find_classes(root)
        self.samples = self._make_dataset(root, class_to_idx)
        
        if not self.samples:
            raise RuntimeError(f"Found 0 files in subfolders of: {root}")

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in self.samples]

    @staticmethod
    def _find_classes(directory: str):
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def _make_dataset(directory, class_to_idx):
        images = []
        for class_name in sorted(class_to_idx.keys()):
            class_path = os.path.join(directory, class_name)
            class_idx = class_to_idx[class_name]
            for filename in sorted(os.listdir(class_path)):
                path = os.path.join(class_path, filename)
                images.append((path, class_idx))
        return images

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)


# =============================================================================
# 4. 统一的数据加载入口函数
# =============================================================================
def get_dataloaders(datasets_dir: str,
                    imagenet_mode: str,
                    batch_size: int = 64,
                    num_workers: int = 8):
    """
    根据指定的模式加载数据集并返回 DataLoader。

    :param datasets_dir: 数据集根目录.
    :param imagenet_mode: 加载模式, 'tiny' 或 'normal'.
    :param batch_size: 批次大小.
    :param num_workers: 工作线程数.
    :return: (train_loader, val_loader, test_loader)
    """
    if imagenet_mode not in ['tiny', 'normal']:
        raise ValueError("imagenet_mode 必须是 'tiny' 或 'normal'")

    logger.info(f"开始加载数据集，模式: {imagenet_mode}")

    # --- 定义通用的图像变换 ---
    transforms_train = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        transforms.RandomErasing(p=0.5, scale=(0.06, 0.08), ratio=(1, 3), value=0)
    ])
    
    
    transforms_val_test = transforms.Compose([
        transforms.Resize(248), 
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])

    # --- 根据模式选择加载逻辑 ---
    if imagenet_mode == 'tiny':
        print("是否正确加载 ImageNet_200 模型")
        # Tiny ImageNet 的加载逻辑
        DIR_TRAIN = os.path.join(datasets_dir, "train/")
        DIR_VAL = os.path.join(datasets_dir, "val/")
        DIR_TEST = os.path.join(datasets_dir, "test/")

        train_labels_list = os.listdir(DIR_TRAIN)
        encoder_labels = LabelEncoder()
        encoder_labels.fit(train_labels_list)

        files_train = [os.path.join(DIR_TRAIN, str(label), "images", str(filename))
                       for label in train_labels_list
                       for filename in os.listdir(os.path.join(DIR_TRAIN, str(label), "images"))]
        
        labels_train = [label
                        for label in train_labels_list
                        for _ in os.listdir(os.path.join(DIR_TRAIN, str(label), "images"))]

        val_images_dir = os.path.join(DIR_VAL, "images")
        files_val = [os.path.join(val_images_dir, f) for f in os.listdir(val_images_dir)]        
        val_df = pd.read_csv(os.path.join(DIR_VAL, "val_annotations.txt"), 
                                    sep="\t", 
                                    header=None,         # <--- 新增：明确没有表头
                                    usecols=[0, 1],      # <--- 新增：只读取前两列
                                    names=["File", "Label"]) # <--- 现在names和usecols匹配        val_labels_map = val_df.set_index("File")["Label"].to_dict()
        
        val_labels_map = val_df.set_index("File")["Label"].to_dict()

        # 创建空的列表来存放有效的文件和标签
        files_val = []
        labels_val = []

        # 遍历注释文件中的每一个条目
        for filename, label in val_labels_map.items():
            image_path = os.path.join(DIR_VAL, "images", filename)
            # 确保这个文件真实存在于硬盘上
            if os.path.exists(image_path):
                files_val.append(image_path)
                labels_val.append(label)
        
        
        test_images_dir = os.path.join(DIR_TEST, "images")
        files_test = sorted([os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir)])
        
        train_dataset = TinyImageNetDataset(files_train, labels_train, encoder_labels, transforms_train, "train")
        val_dataset = TinyImageNetDataset(files_val, labels_val, encoder_labels, transforms_val_test, "val")
        test_dataset = TinyImageNetDataset(files_test, None, None, transforms_val_test, "test")

    else: # imagenet_mode == 'normal'
        # 标准 ImageFolder 结构的加载逻辑
        DIR_TRAIN = os.path.join(datasets_dir, 'train')
        DIR_VAL = os.path.join(datasets_dir, 'val_mini') # 假设验证集也是标准结构
        
        train_dataset = StandardImageFolder(root=DIR_TRAIN, transform=transforms_train)
        val_dataset = StandardImageFolder(root=DIR_VAL, transform=transforms_val_test)
        
        # 标准模式下，测试集通常没有标签，且结构可能不同，这里简化处理
        # 如果有标准测试集，可以类似地进行加载
        test_dataset = None # 或者根据实际情况加载

    # --- 创建 DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None
        
    logger.info(f"数据加载完成。训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# 5. 主程序入口，演示如何使用
# =============================================================================
if __name__ == "__main__":
    
    # 定义你要扫描的目录。'.' 表示当前目录。
    # 你可以把它改成任何你想要的路径，例如 "C:/Users/YourUser/Documents"
    target_directory = '/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/calibration_dataset/resnet18_calib/input.1' 
    
    # 定义输出文件的名字。
    output_filename = '/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/dipoorlet_log/3_dipoorlet_models_resnet18/resnet18_calib.txt'
    
    # 调用函数来执行扫描和写入操作。
    scan_and_write_paths(target_directory, output_filename)
    
    sys.exit()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- 示例1: 加载 Tiny ImageNet 数据集 ---
    print("\n" + "="*50)
    print("演示加载 Tiny ImageNet ('tiny' 模式)")
    print("="*50)
    tiny_imagenet_path = "/mnt/share_disk/bruce_trie/outputs/tiny-imagenet-200/"
    if os.path.exists(tiny_imagenet_path):
        try:
            train_loader_tiny, val_loader_tiny, test_loader_tiny = get_dataloaders(
                datasets_dir=tiny_imagenet_path,
                imagenet_mode='tiny',
                batch_size=16
            )
            # 验证一下数据
            images, labels = next(iter(val_loader_tiny))
            print(f"Tiny ImageNet 验证集 - 一个批次的数据形状: {images.shape}")
            print(f"Tiny ImageNet 验证集 - 一个批次的标签: {labels}")
        except Exception as e:
            print(f"加载 'tiny' 模式失败: {e}")
    else:
        print(f"路径不存在，跳过 'tiny' 模式演示: {tiny_imagenet_path}")
        
    # --- 示例2: 加载标准 ImageNet/ImageFolder 结构的数据集 ---
    print("\n" + "="*50)
    print("演示加载标准 ImageFolder ('normal' 模式)")
    print("="*50)
    
    # 请将此路径替换为您的标准ImageNet或类似ImageFolder结构的数据集路径
    # 例如：/path/to/your/dataset/
    #         ├── train/
    #         │   ├── class_a/
    #         │   │   ├── xxx.png
    #         │   │   └── xxy.png
    #         │   └── class_b/
    #         │       ├── ...
    #         └── val/
    #             ├── class_a/
    #             │   ├── ...
    #             └── class_b/
    #                 ├── ...
    normal_imagenet_path = "/mnt/share_disk/bruce_trie/imagenet/"
    if os.path.exists(normal_imagenet_path):
        try:
            train_loader_normal, val_loader_normal, _ = get_dataloaders(
                datasets_dir=normal_imagenet_path,
                imagenet_mode='normal',
                batch_size=16
            )
            images, labels = next(iter(val_loader_normal))
            print(f"Normal ImageNet 验证集 - 一个批次的数据形状: {images.shape}")
            print(f"Normal ImageNet 验证集 - 一个批次的标签: {labels}")
        except Exception as e:
            print(f"加载 'normal' 模式失败: {e}")
    else:
        print(f"路径不存在，跳过 'normal' 模式演示: {normal_imagenet_path}")
        
        
        



