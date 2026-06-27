import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from autoattack import AutoAttack


class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y, epsilon=0.0314, k=20, alpha=0.00784):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x


class LinfPGDTAttack(object):
    def __init__(self, model):
        self.model = model

    def other_target(self, y):
        num_classes = 10
        new_y = torch.empty_like(y)
        for i in range(y.size(0)):
            # 获取除 y[i] 外的所有类别
            others = [c for c in range(num_classes) if c != y[i].item()]
            # 随机选择一个
            new_y[i] = torch.tensor(np.random.choice(others), device=y.device)
        return new_y

    def perturb(self, x_natural, y, epsilon=0.0314, k=20, alpha=0.00784):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        y = self.other_target(y)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() - alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

class LinfPGDHAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y, epsilon=0.0314, k=20, alpha=0.00784):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits[:,:-1], y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def get_cw(model, x, y, device, epsilon = 8.0/255, step_size = 2.0/255, num_class = 10):
    model.eval()
    x_adv = Variable(x.data, requires_grad=True)
    random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-epsilon, epsilon).to(device)
    x_adv = Variable(x_adv.data + random_noise, requires_grad=True)
    # onehot_targets = torch.eye(args.num_classes)[y].to(device)
    onehot_targets = torch.nn.functional.one_hot(torch.tensor(y), num_classes=num_class).to(torch.float).cuda(non_blocking=True)
    for _ in range(100):
        opt = optim.SGD([x_adv], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            logits = model(x_adv)

            self_loss = torch.sum(onehot_targets * logits, dim=1)
            other_loss = torch.max((1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

            loss = -torch.sum(torch.clamp(self_loss - other_loss + 50, 0))
            loss = loss / onehot_targets.shape[0]

        loss.backward()
        eta = step_size * x_adv.grad.data.sign()
        x_adv = Variable(x_adv.data + eta, requires_grad=True)
        eta = torch.clamp(x_adv.data - x.data, -epsilon, epsilon)
        x_adv = Variable(x.data + eta, requires_grad=True)
        x_adv = Variable(torch.clamp(x_adv, 0, 1.0), requires_grad=True)
    return x_adv

class NONEAttack:
    def __init__(self, model, num_class):
        self.model = model
    def attack(self, inputs, targets):
        return inputs

class FGSMAttack:
    def __init__(self, model, num_class):
        self.model = model
        self.adversary = LinfPGDAttack(self.model)
    def attack(self, inputs, targets):
        return self.adversary.perturb(inputs, targets, epsilon= 8./255, k=1, alpha= 8./255)

class PGDAttack:
    def __init__(self, model, num_class):
        self.model = model
        self.adversary = LinfPGDAttack(self.model)
    def attack(self, inputs, targets):
        return self.adversary.perturb(inputs, targets, epsilon= 8./255, k=20, alpha= 2./255)

class PGDTAttack:
    def __init__(self, model, num_class):
        self.model = model
        self.adversary = LinfPGDTAttack(self.model)
    def attack(self, inputs, targets):
        return self.adversary.perturb(inputs, targets, epsilon= 8./255, k=20, alpha= 2./255)

class PGDHAttack:
    def __init__(self, model, num_class):
        self.model = model
        self.adversary = LinfPGDHAttack(self.model)
    def attack(self, inputs, targets):
        return self.adversary.perturb(inputs, targets, epsilon= 8./255, k=20, alpha= 2./255)

class CWAttack:
    def __init__(self, model, num_class):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_class = num_class
    def attack(self, inputs, targets):
        return get_cw(self.model, inputs, targets, self.device, epsilon = 8.0/255, step_size = 2.0/255, num_class = self.num_class)
class AAAttack:
    def __init__(self, model, num_class):
        self.model = model
        self.adversary = AutoAttack(self.model, norm="Linf", eps=8.0/255, verbose = False)
        self.adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    def attack(self, inputs, targets):
        return self.adversary.run_standard_evaluation(inputs, targets, bs=inputs.shape[0])
class SQUAREAttack:
    def __init__(self, model, num_class):
        self.model = model
        self.adversary = AutoAttack(self.model, norm="Linf", eps=8.0/255, verbose = False)
        self.adversary.attacks_to_run = ['square']
        self.adversary.square.n_queries = 1000

    def attack(self, inputs, targets):
        return self.adversary.run_standard_evaluation(inputs, targets, bs=inputs.shape[0])

class Evaluate:
    def __init__(self, model, num_class = 10, custom_attacks = None):
        # 支持自定义攻击列表；未指定时使用默认全部攻击
        if custom_attacks is not None:
            self.attack_list = list(custom_attacks)
        else:
            self.attack_list = ["NONE", "PGD", "CW", "SQUARE", "AA"]
        self.model = model
        for attack in self.attack_list:
            attack_class_name = attack + "Attack"
            try:
                self.__dict__[attack] = eval(attack_class_name)(self.model, num_class)
            except NameError:
                print(f"[WARNING] 攻击类型 {attack_class_name} 不存在，跳过")
    def attack(self, inputs, targets, attack):
        return self.__dict__[attack].attack(inputs, targets)
    def getList(self):
        return self.attack_list