import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNet18
class MergeModel(nn.Module):
    def __init__(self, model_list):
        super(MergeModel, self).__init__()
        self.model_list = model_list
        self.length = len(model_list)
    def forward(self, x, loss = "softmax_logit", calculate = "average"):
        # numbers = random.sample(range(self.length), 1)
        outputs = []
        for number in range(self.length):
            if loss == "softmax_logit":
                outputs.append(F.softmax(self.model_list[number](x), dim=1))
        if calculate == "average":
            ave_output = torch.stack(outputs).mean(dim=0)
            return ave_output
        


def load_model(model: str, dataset: str, basic_model, adversarial_defenses):
    """
    尝试加载预训练权重；若文件不存在则回退到随机初始化。
    权重路径: {dataset}_{model}_dwe
    """
    # 加载集成网络
    model_list = []
    for i in range(10):
        net = ResNet18(num_classes=10)
        net = net.cuda().eval()
        weight_path = f'/home/ubuntu/zengyi/adversarial_system-master/backend/checkpoint/dwe/{dataset}_{model}_{i}'
        try:
            net.load_state_dict(torch.load(weight_path, map_location='cpu'))
        except FileNotFoundError:
            print(f"[WARNING] 权重文件 {weight_path} 不存在，使用随机初始化")
        except Exception as e:
            print(f"[WARNING] 加载权重失败 ({e})，使用随机初始化")
        net.eval()
        model_list.append(net)
    if adversarial_defenses:
        model_list[-1] = basic_model
    mergeModel = MergeModel(model_list)
    mergeModel.eval()
    return mergeModel
