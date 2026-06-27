import torch
from vgg16 import VGG16
from resnet import ResNet18
from wideresnet import WideResNet


def _get_num_classes(dataset: str) -> int:
    mapping = {
        "cifar10": 10, "cifar100": 100,
        "stl": 10, "imagenet": 200, "svhn": 10,
    }
    return mapping.get(dataset, 10)


def _create_model(model: str, num_classes: int):
    """只创建模型结构，不加载权重"""
    if model == "resnet18":
        return ResNet18(num_classes)
    elif model == "vgg16":
        return VGG16(num_classes)
    elif model == "wrn28":
        return WideResNet(28, num_classes, widen_factor=10, dropRate=0.0)
    else:
        raise ValueError(f"未知模型: {model}")


def load_model_structure(model: str, dataset: str, purification: bool = False):
    """
    只返回模型结构（随机初始化），不加载预训练权重。
    用于没有权重文件的场景。
    """
    num_classes = _get_num_classes(dataset)
    if purification:
        num_classes = num_classes + 1
    net = _create_model(model, num_classes)
    net.eval()
    return net


def load_model(model: str, dataset: str, purification: bool = False):
    """
    尝试加载预训练权重；若文件不存在则回退到随机初始化。
    权重路径: {dataset}_{model}_basic.pth
    """
    num_classes = _get_num_classes(dataset)
    if purification:
        num_classes = num_classes + 1
    net = _create_model(model, num_classes)

    weight_path = f'{dataset}_{model}_basic.pth'
    try:
        net.load_state_dict(torch.load(weight_path, map_location='cpu'))
    except FileNotFoundError:
        print(f"[WARNING] 权重文件 {weight_path} 不存在，使用随机初始化")
    except Exception as e:
        print(f"[WARNING] 加载权重失败 ({e})，使用随机初始化")

    net.eval()
    return net
