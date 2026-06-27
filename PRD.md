# 对抗防御工作流系统 PRD

## 核心功能

### 工作流
- 数据集卡片：可选CIFAR-10，CIFAR-100、ImageNet、STL
- 模型卡片：ResNet-18、WideResNet-28-10
- 攻击卡片：PGD、FGSM、AutoAttack
- 防御卡片:
1. 输入重构卡片（只能介于模型卡片之前）：Diffpure、EBM、IEDN
2. 对抗训练卡片（只能作用于模型卡片）：PGD-AT、TRADES
3. 主动防御卡片（只能作用于模型卡片）：AHD
4. 集成防御卡片（只能作用于模型卡片）：DWE


### 前端页面
- 工作流界面
- 可拖拽的各种卡片
