import type { NodeData } from '../types'

/**
 * New simplified connection rules:
 *
 *   Dataset             ──► Attack
 *   Dataset             ──► InputReconstruction Defense
 *   Dataset             ──► Model
 *   Dataset             ──► AdversarialTraining / EnsembleDefense
 *
 *   Attack              ──► InputReconstruction Defense
 *   Attack              ──► Model
 *   Attack              ──► AdversarialTraining / EnsembleDefense
 *
 *   ActiveDefense       ──► Attack
 *
 *   InputReconstruction ──► Model
 *   AdversarialTraining ──► Model
 *   EnsembleDefense     ──► Model
 *
 *   Model ──► (nothing — terminal)
 */
export function isValidConnection(
  source: NodeData,
  target: NodeData,
): boolean {
  if (source.cardId === target.cardId) return false

  const srcCat = source.category
  const tgtCat = target.category
  const srcDef = source.defenseSubtype
  const tgtDef = target.defenseSubtype

  // ---- Rules by target category ----

  // 攻击: 可以被数据集和主动防御连接
  if (tgtCat === 'attack') {
    if (srcCat === 'dataset') return true
    if (srcCat === 'defense' && srcDef === 'active_defense') return true
    return false
  }

  // 输入重构: 只能被攻击和数据集连接
  if (tgtCat === 'defense' && tgtDef === 'input_reconstruction') {
    return srcCat === 'dataset' || srcCat === 'attack'
  }

  // 模型: 可以被数据集、攻击、输入重构、集成防御、对抗训练连接（主动防御除外）
  if (tgtCat === 'model') {
    if (srcCat === 'dataset' || srcCat === 'attack') return true
    if (srcCat === 'defense' && srcDef !== 'active_defense') return true
    return false
  }

  // 结果: 只能被模型连接
  if (tgtCat === 'result') {
    return srcCat === 'model'
  }

  // 主动防御: 只能连接到攻击，不接受输入
  if (tgtCat === 'defense' && tgtDef === 'active_defense') {
    return false
  }

  // 其他防御（对抗训练、集成防御）: 可以被数据集和攻击连接
  if (tgtCat === 'defense' && tgtDef !== 'input_reconstruction') {
    return srcCat === 'dataset' || srcCat === 'attack'
  }

  // Model cannot connect to anything
  if (srcCat === 'model') return false

  return false
}

/**
 * Returns a human-readable reason why the connection is invalid.
 */
export function getConnectionError(
  source: NodeData,
  target: NodeData,
): string | null {
  if (isValidConnection(source, target)) return null

  const srcCat = source.category
  const tgtCat = target.category
  const tgtDef = target.defenseSubtype

  // Model as source
  if (srcCat === 'model') {
    return '模型不能作为连线起点'
  }

  // Attack as target
  if (tgtCat === 'attack') {
    return '攻击只能被数据集或主动防御连接'
  }

  // InputRecon as target
  if (tgtCat === 'defense' && tgtDef === 'input_reconstruction') {
    return '输入重构只能被数据集或攻击连接'
  }

  // Model as target
  if (tgtCat === 'model') {
    return '模型只能被数据集、攻击、输入重构、集成防御、对抗训练连接'
  }

  // Result as target (srcCat !== 'model' already handled above)
  if (tgtCat === 'result') {
    return '结果卡片只能被模型连接'
  }

  // ActiveDefense — only outputs, cannot receive connections
  if (tgtCat === 'defense' && tgtDef === 'active_defense') {
    return '主动防御不接受输入连接，只能连接到攻击卡片'
  }

  // Other defenses as target
  if (tgtCat === 'defense') {
    return '该防御只能被数据集或攻击连接'
  }

  return `无效连接: ${srcCat} → ${tgtCat}`
}
