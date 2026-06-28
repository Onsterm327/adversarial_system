// ============================================================
// Card type definitions for the adversarial defense workflow
// ============================================================

export type CardCategory = 'dataset' | 'model' | 'attack' | 'defense' | 'result'

export type DefenseSubtype =
  | 'input_reconstruction'
  | 'adversarial_training'
  | 'active_defense'
  | 'ensemble_defense'

/** Static definition of a card in the palette */
export interface CardDef {
  id: string
  name: string
  category: CardCategory
  defenseSubtype?: DefenseSubtype
  description: string
  /** Attack cards: which attacks are available to add */
  availableAttacks?: string[]
}

/** Runtime data stored on each React Flow node */
export interface NodeData {
  cardId: string
  label: string
  category: CardCategory
  defenseSubtype?: DefenseSubtype
  /** Attack nodes: which attacks can be added */
  availableAttacks?: string[]
  /** Attack nodes: list of selected attack names (e.g. ["Clean", "PGD"]) */
  selectedAttacks?: string[]
  [key: string]: unknown  // satisfy Record<string, unknown> for React Flow
}

/** Category display metadata */
export interface CategoryMeta {
  label: string
  color: string
  emoji: string
}

export const CATEGORY_META: Record<CardCategory, CategoryMeta> = {
  dataset:  { label: '数据集',   color: '#3b82f6', emoji: '📊' },
  model:    { label: '模型',     color: '#22c55e', emoji: '🧠' },
  attack:   { label: '攻击',     color: '#ef4444', emoji: '⚔️' },
  defense:  { label: '防御',     color: '#a855f7', emoji: '🛡️' },
  result:   { label: '结果',     color: '#f59e0b', emoji: '📋' },
}

export const DEFENSE_SUBTYPE_LABELS: Record<DefenseSubtype, string> = {
  input_reconstruction: '输入重构',
  adversarial_training: '对抗训练',
  active_defense:       '主动防御',
  ensemble_defense:     '集成防御',
}
