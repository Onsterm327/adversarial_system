import torch
from .resnet import ResNet18
class MergeModel(nn.Module):
    def __init__(self, model_list):
        super(MergeModel, self).__init__()
        self.model_list = model_list
        self.length = len(model_list)
    def forward(self, x, loss = "softmax_logit", calculate = "average", whole = False):
        # numbers = random.sample(range(self.length), 1)
        if whole:
            numbers = [0,1,2,3,4,5,6,7,8,9]
        else:
            numbers = [9]
        outputs = []
        for number in numbers:
            if loss == "softmax_logit":
                outputs.append(F.softmax(self.model_list[number](x), dim=1))
            elif loss == "log_softmax_logit":
                outputs.append(F.log_softmax(self.model_list[number](x), dim=1))
            elif loss == "logit":
                outputs.append(self.model_list[number](x))
        if calculate == "average":
            ave_output = torch.stack(outputs).mean(dim=0)
            return ave_output
        elif calculate == "vote":
            return self.vote(outputs)
        elif calculate == "attention":
            stacked_logits = torch.stack(outputs, dim=1)
            # print("attention")
            Q = stacked_logits  # [B, M, C]
            K = stacked_logits  # [B, M, C]
            V = stacked_logits  # [B, M, C]
            attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (10 ** 0.5)
            # softmax获取注意力权重
            attn_weights = F.softmax(attn_scores, dim=-1)  # [B, M, M]
            # print(attn_weights)
            
            # 应用注意力
            attended = torch.matmul(attn_weights, V)  # [B, M, C]
            # print(attended.shape)
            
            # 平均所有模型的输出
            combined = attended.mean(dim=1)
            return combined
    def vote(self, outputs):
        predicteds = torch.stack([output.max(1)[1] for output in outputs], dim=0)
        values, _ = torch.mode(predicteds, dim=0)
        return F.one_hot(values, num_classes=10).float()
    



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
    权重路径: {dataset}_{model}_dwe
    """

    model_list = []
    for i in range(10):
        net = ResNet18(num_classes=10)
        net = net.to()
        weight_path = f'{dataset}_{model}_dwe_{i}'
        try:
            net.load_state_dict(torch.load(weight_path, map_location='cpu'))
        except FileNotFoundError:
            print(f"[WARNING] 权重文件 {weight_path} 不存在，使用随机初始化")
        except Exception as e:
            print(f"[WARNING] 加载权重失败 ({e})，使用随机初始化")
        net.eval()
        model_list.append(net)
    mergeModel = MergeModel(model_list)
    mergeModel.eval()
    return mergeModel
