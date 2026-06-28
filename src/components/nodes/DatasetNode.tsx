import { Handle, Position, type NodeProps } from '@xyflow/react'
import type { NodeData } from '../../types'
import { CATEGORY_META } from '../../types'
import './CardNode.css'

export default function DatasetNode({ data }: NodeProps) {
  const nodeData = data as unknown as NodeData
  const meta = CATEGORY_META[nodeData.category]

  return (
    <div className="card-node" data-category={nodeData.category}>
      <Handle type="source" position={Position.Right} />
      <div className="card-header">
        <span className="card-emoji">{meta.emoji}</span>
        <span className="card-name">{meta.label}</span>
      </div>
      <div className="card-tags">
        <span className="card-tag" style={{ background: meta.color }}>{nodeData.label}</span>
      </div>
    </div>
  )
}
