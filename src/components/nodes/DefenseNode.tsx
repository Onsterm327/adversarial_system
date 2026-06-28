import { Handle, Position, type NodeProps } from '@xyflow/react'
import type { NodeData } from '../../types'
import { CATEGORY_META, DEFENSE_SUBTYPE_LABELS } from '../../types'
import './CardNode.css'

const DEFENSE_COLORS: Record<string, string> = {
  input_reconstruction: '#8b5cf6',
  adversarial_training: '#ec4899',
  active_defense:       '#f97316',
  ensemble_defense:     '#14b8a6',
}

const OUTPUT_ONLY_SUBTYPES = new Set([
  'adversarial_training',
  'active_defense',
  'ensemble_defense',
])

export default function DefenseNode({ data }: NodeProps) {
  const nodeData = data as unknown as NodeData
  const meta = CATEGORY_META[nodeData.category]
  const subLabel = nodeData.defenseSubtype
    ? DEFENSE_SUBTYPE_LABELS[nodeData.defenseSubtype]
    : meta.label
  const isOutputOnly = nodeData.defenseSubtype
    ? OUTPUT_ONLY_SUBTYPES.has(nodeData.defenseSubtype)
    : false
  const tagColor = nodeData.defenseSubtype
    ? DEFENSE_COLORS[nodeData.defenseSubtype] ?? meta.color
    : meta.color

  return (
    <div className="card-node" data-category={nodeData.category}>
      {!isOutputOnly && (
        <Handle type="target" position={Position.Left} />
      )}
      <Handle type="source" position={Position.Right} />
      <div className="card-header">
        <span className="card-emoji">{meta.emoji}</span>
        <span className="card-name">{subLabel}</span>
      </div>
      <div className="card-tags">
        <span className="card-tag" style={{ background: tagColor }}>{nodeData.label}</span>
      </div>
    </div>
  )
}
