import type { CardDef } from '../types'

export const CARDS: CardDef[] = [
  // ---- Datasets ----
  { id: 'cifar10',  name: 'CIFAR-10',  category: 'dataset', description: '10 类彩色图像，60,000 张 32×32' },
  { id: 'cifar100', name: 'CIFAR-100', category: 'dataset', description: '100 类彩色图像，60,000 张 32×32' },
  { id: 'imagenet', name: 'ImageNet',  category: 'dataset', description: '1,000 类高清图像，120 万+训练样本' },
  { id: 'stl',      name: 'STL',       category: 'dataset', description: '10 类 96×96 图像，无监督学习基准' },

  // ---- Models ----
  { id: 'resnet18',       name: 'ResNet-18',        category: 'model', description: '18 层残差网络' },
  { id: 'wideresnet2810', name: 'WideResNet-28-10', category: 'model', description: '宽度因子 10，深度 28 的 Wide ResNet' },

  // ---- Attacks ----
  { id: 'adaptive_attack',  name: '自适应攻击', category: 'attack', availableAttacks: ['AutoAttack', 'PGD', 'FGSM'], description: '自适应攻击' },
  { id: 'adversary_attack', name: '对抗攻击',   category: 'attack', availableAttacks: ['AutoAttack', 'PGD', 'FGSM'], description: '对抗攻击' },

  // ---- Defenses: Input Reconstruction ----
  { id: 'diffpure', name: 'DiffPure', category: 'defense', defenseSubtype: 'input_reconstruction', description: '扩散模型去噪' },
  { id: 'ebm',      name: 'EBM',      category: 'defense', defenseSubtype: 'input_reconstruction', description: '能量模型防御' },
  { id: 'iedn',     name: 'IEDN',     category: 'defense', defenseSubtype: 'input_reconstruction', description: '图像增强去噪网络' },

  // ---- Defenses: Adversarial Training ----
  { id: 'pgd_at', name: 'PGD-AT', category: 'defense', defenseSubtype: 'adversarial_training', description: 'PGD 对抗训练' },
  { id: 'trades', name: 'TRADES', category: 'defense', defenseSubtype: 'adversarial_training', description: '鲁棒性权衡对抗训练' },

  // ---- Defenses: Active Defense ----
  { id: 'ahd', name: 'AHD', category: 'defense', defenseSubtype: 'active_defense', description: '主动混合防御' },

  // ---- Defenses: Ensemble Defense ----
  { id: 'dwe', name: 'DWE', category: 'defense', defenseSubtype: 'ensemble_defense', description: '多样性加权集成' },

  // ---- Result ----
  { id: 'result', name: '结果', category: 'result', description: '执行并查看链路结果' },
]

/** Group cards by category (and defense subtype for defense cards) */
export function getCardsByCategory() {
  const datasets  = CARDS.filter(c => c.category === 'dataset')
  const models    = CARDS.filter(c => c.category === 'model')
  const attacks   = CARDS.filter(c => c.category === 'attack')
  const defenses  = CARDS.filter(c => c.category === 'defense')

  // Group defenses by subtype
  const defenseGroups = new Map<string, CardDef[]>()
  for (const d of defenses) {
    const key = d.defenseSubtype ?? 'other'
    if (!defenseGroups.has(key)) defenseGroups.set(key, [])
    defenseGroups.get(key)!.push(d)
  }

  return { datasets, models, attacks, defenseGroups, result: CARDS.find(c => c.category === 'result') }
}

export function getCardById(id: string): CardDef | undefined {
  return CARDS.find(c => c.id === id)
}
