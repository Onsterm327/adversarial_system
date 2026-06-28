import { Handle, Position, type NodeProps } from '@xyflow/react'
import type { NodeData } from '../../types'
import { CATEGORY_META, DEFENSE_SUBTYPE_LABELS } from '../../types'
import './CardNode.css'

const DEFENSE_SUBTYPE_COLORS: Record<string, string> = {
  input_reconstruction: '#8b5cf6',
  adversarial_training: '#ec4899',
  active_defense:       '#f97316',
  ensemble_defense:     '#14b8a6',
}

/** Subtypes that only have a right source handle (no inputs) */
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
    : '防御'

  const isOutputOnly = nodeData.defenseSubtype
    ? OUTPUT_ONLY_SUBTYPES.has(nodeData.defenseSubtype)
    : false

  return (
    <div className="card-node" data-category={nodeData.category}>
      {!isOutputOnly && (
        <>
          <Handle type="target" position={Position.Left} isConnectableStart={false} isConnectableEnd={true} />
          <Handle type="target" position={Position.Top} isConnectableStart={false} isConnectableEnd={true} />
          <Handle type="target" position={Position.Bottom} isConnectableStart={false} isConnectableEnd={true} />
        </>
      )}
      <Handle type="source" position={Position.Right} isConnectableStart={true} isConnectableEnd={false} />
      <div className="card-header">
        <span className="card-emoji">{meta.emoji}</span>
        <span className="card-name">{nodeData.label}</span>
      </div>
      <div className="card-badges">
        <span className="badge" style={{ background: meta.color }}>{meta.label}</span>
        {nodeData.defenseSubtype && (
          <span
            className="badge"
            style={{ background: DEFENSE_SUBTYPE_COLORS[nodeData.defenseSubtype] ?? '#a855f7' }}
          >
            {subLabel}
          </span>
        )}
      </div>
      <div className="card-desc">
        {isOutputOnly
          ? '仅输出 — 连接到模型'
          : '输入重构 — 左/上/下输入，右侧输出'}
      </div>
    </div>
  )
}
