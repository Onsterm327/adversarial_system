from torchvision.datasets import STL10
import torchvision
import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image

class STL10WithIndex(STL10):
    def __init__(self, *args, **kwargs):
        super(STL10WithIndex, self).__init__(*args, **kwargs)
        self.targets = torch.from_numpy(self.labels).long()
    def __getitem__(self, index):
        # 调用父类方法获取图像和标签
        img, label = super().__getitem__(index)
        # 返回图像、标签和索引
        return img, label


class TinyImageNet(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.classes = []
        self.class_to_idx = {}
        self.images = []
        self.targets = []
        # 构建类别索引
        with open(os.path.join(root_dir, 'wnids.txt'), 'r') as f:
            for i, line in enumerate(f):
                class_name = line.strip()
                self.class_to_idx[class_name] = i
                self.classes.append(class_name)
        
        # 加载图像路径和标签
        if split == 'train':
            for class_name in self.classes:
                class_dir = os.path.join(root_dir, 'train', class_name, 'images')
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.JPEG'):
                        self.images.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
                        self.targets.append(self.class_to_idx[class_name])
        elif split == 'val':
            with open(os.path.join(root_dir, 'val', 'val_annotations.txt'), 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name, class_name = parts[0], parts[1]
                    self.images.append((os.path.join(root_dir, 'val', 'images', img_name), 
                                      self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_dataset(dataset, batchSize = 100):
    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='/home/ubuntu/zengyi/lwf-AT/data/cifar10', train=True, download=False, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='/home/ubuntu/zengyi/lwf-AT/data/cifar10', train=False, download=False, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=True, num_workers=4)
    elif dataset == "cifar100":
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])

        train_dataset = torchvision.datasets.CIFAR100(root='/home/ubuntu/zengyi/lwf-AT/diffusion/data/cifar100', train=True, download=False, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root='/home/ubuntu/zengyi/lwf-AT/diffusion/data/cifar100', train=False, download=False, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=True, num_workers=4)
    elif dataset == "imagenet":
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            # transforms.Resize(64),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = TinyImageNet(root_dir='/home/ubuntu/zengyi/lwf-AT/diffusion/data/tiny-imagenet-200', split='train', transform=transform_train)
        test_dataset = TinyImageNet(root_dir='/home/ubuntu/zengyi/lwf-AT/diffusion/data/tiny-imagenet-200', split='val', transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=True, num_workers=4)
    elif dataset == "stl":
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            # transforms.Resize(64),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = STL10WithIndex(
            root='/home/ubuntu/zengyi/lwf-AT/diffusion/data',
            split='train',
            download=False,
            transform=transform_train
        )
        test_dataset = STL10WithIndex(
            root='/home/ubuntu/zengyi/lwf-AT/diffusion/data',
            split='test',
            download=False,
            transform=transform_test
        )
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size = batchSize, shuffle = True, num_workers = 4)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=4)
    elif dataset == "svhn":
        learning_rate = 0.01
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.ToTensor(),
        ])
        # 测试数据集不用做数据增强
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.SVHN(root = '/home/ubuntu/zengyi/lwf-AT/diffusion/data/svhn', split='train', download = True, transform = transform_train)
        test_dataset  = torchvision.datasets.SVHN(root = '/home/ubuntu/zengyi/lwf-AT/diffusion/data/svhn', split='test', download = True, transform = transform_test)
        
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size = batchSize, shuffle = True, num_workers = 4)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=4)
    return test_loader, train_loader