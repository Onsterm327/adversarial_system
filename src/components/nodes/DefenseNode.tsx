import { useCallback } from 'react'
import { Handle, Position, useReactFlow, useStore, type NodeProps } from '@xyflow/react'
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

export default function DefenseNode({ id, data }: NodeProps) {
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
  const isEnsemble = nodeData.defenseSubtype === 'ensemble_defense'
  const atkValue = (nodeData.defenseParams?.atk as number) ?? 2

  const { setNodes } = useReactFlow()
  const allNodes = useStore(s => s.nodes)

  const handleAtkChange = useCallback(
    (val: number) => {
      setNodes(
        allNodes.map(n =>
          n.id === id
            ? { ...n, data: { ...n.data, defenseParams: { ...(n.data as NodeData).defenseParams, atk: val } } }
            : n,
        ),
      )
    },
    [id, setNodes, allNodes],
  )

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

      {/* DWE slider */}
      {isEnsemble && (
        <div className="defense-slider-row">
          <label className="defense-slider-label">
            atk: <strong>{atkValue}</strong>
          </label>
          <input
            type="range"
            className="defense-slider"
            min={1}
            max={5}
            step={1}
            value={atkValue}
            onChange={e => handleAtkChange(Number(e.target.value))}
          />
        </div>
      )}
    </div>
  )
}
