import { Handle, Position, type NodeProps } from '@xyflow/react'
import type { NodeData } from '../../types'
import { CATEGORY_META } from '../../types'
import './CardNode.css'

export default function ModelNode({ data }: NodeProps) {
  const nodeData = data as unknown as NodeData
  const meta = CATEGORY_META[nodeData.category]

  return (
    <div className="card-node" data-category={nodeData.category}>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
      <div className="card-header">
        <span className="card-emoji">{meta.emoji}</span>
        <span className="card-name">{nodeData.label}</span>
      </div>
      <div className="card-badges">
        <span className="badge" style={{ background: meta.color }}>{meta.label}</span>
      </div>
    </div>
  )
}
