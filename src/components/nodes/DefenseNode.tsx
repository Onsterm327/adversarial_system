import { Handle, Position, type NodeProps } from '@xyflow/react'
import type { NodeData } from '../../types'
import { CATEGORY_META } from '../../types'
import './CardNode.css'

/** Subtypes that only have a source handle (no input) */
const OUTPUT_ONLY_SUBTYPES = new Set([
  'adversarial_training',
  'active_defense',
  'ensemble_defense',
])

export default function DefenseNode({ data }: NodeProps) {
  const nodeData = data as unknown as NodeData
  const meta = CATEGORY_META[nodeData.category]

  const isOutputOnly = nodeData.defenseSubtype
    ? OUTPUT_ONLY_SUBTYPES.has(nodeData.defenseSubtype)
    : false

  return (
    <div className="card-node" data-category={nodeData.category}>
      {!isOutputOnly && (
        <Handle type="target" position={Position.Left} />
      )}
      <Handle type="source" position={Position.Right} />
      <div className="card-header">
        <span className="card-emoji">{meta.emoji}</span>
        <span className="card-name">{nodeData.label}</span>
      </div>
    </div>
  )
}
