import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg16 import VGG16
from .resnet import ResNet18
from .wideresnet import WideResNet

def load_model(model, dataset):
    if dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    elif dataset == "stl":
        num_classes = 10
    elif dataset == "imagenet":
        num_classes = 200
    elif dataset == "svhn":
        num_classes = 10
    
    num_classes = num_classes + 1

    if model == "resnet18":
        net = ResNet18(num_classes)
    elif model == "vgg16":
        net = VGG16(num_classes)
    elif model == "wrn28":
        net = WideResNet(28, num_classes, widen_factor=10, dropRate=0.0)
    
    net.load_state_dict(torch.load(f'{dataset}_{model}_ahd.pth'))
    net.eval()

    return net
