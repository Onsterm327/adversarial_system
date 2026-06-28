import { useCallback, useState } from 'react'
import { Handle, Position, useReactFlow, useStore, type NodeProps } from '@xyflow/react'
import type { NodeData } from '../../types'
import { CATEGORY_META } from '../../types'
import './CardNode.css'

const AVAILABLE_ATTACKS = ['PGD', 'FGSM', 'AutoAttack']
const DEFAULT_ATTACKS = ['Clean']

export default function AttackNode({ id, data }: NodeProps) {
  const nodeData = data as unknown as NodeData
  const meta = CATEGORY_META[nodeData.category]
  const attacks: string[] = nodeData.selectedAttacks ?? DEFAULT_ATTACKS
  const [menuOpen, setMenuOpen] = useState(false)
  const { setNodes } = useReactFlow()
  // Get full node list to properly update
  const allNodes = useStore(s => s.nodes)

  const updateAttacks = useCallback(
    (newAttacks: string[]) => {
      setNodes(
        allNodes.map(n =>
          n.id === id
            ? { ...n, data: { ...n.data, selectedAttacks: newAttacks, label: newAttacks.join(' / ') } }
            : n,
        ),
      )
    },
    [id, setNodes, allNodes],
  )

  const addAttack = useCallback(
    (name: string) => {
      if (!attacks.includes(name)) {
        updateAttacks([...attacks, name])
      }
      setMenuOpen(false)
    },
    [attacks, updateAttacks],
  )

  const removeAttack = useCallback(
    (name: string) => {
      if (name === 'Clean') return // Clean 不可删除
      updateAttacks(attacks.filter(a => a !== name))
    },
    [attacks, updateAttacks],
  )

  const availableToAdd = AVAILABLE_ATTACKS.filter(a => !attacks.includes(a))

  return (
    <div className="card-node attack-node" data-category={nodeData.category}>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />

      <div className="card-header">
        <span className="card-emoji">{meta.emoji}</span>
        <span className="card-name">攻击 ({attacks.length})</span>
      </div>

      <div className="card-badges">
        <span className="badge" style={{ background: meta.color }}>{meta.label}</span>
      </div>

      {/* Attack list */}
      <div className="attack-list">
        {attacks.map(name => (
          <div
            className={`attack-tag ${name}`}
            key={name}
          >
            <span className="attack-tag-name">{name}</span>
            {name !== 'Clean' && (
              <button
                className="attack-tag-remove"
                onClick={() => removeAttack(name)}
                title={`移除 ${name}`}
              >
                ×
              </button>
            )}
          </div>
        ))}
      </div>

      {/* Add attack button + dropdown */}
      <div className="attack-add-row">
        {availableToAdd.length > 0 ? (
          <div className="attack-add-wrapper">
            <button
              className="attack-add-btn"
              onClick={() => setMenuOpen(!menuOpen)}
            >
              + 添加攻击
            </button>
            {menuOpen && (
              <div className="attack-add-menu">
                {availableToAdd.map(name => (
                  <button
                    key={name}
                    className="attack-add-option"
                    onClick={() => addAttack(name)}
                  >
                    {name}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <span className="attack-full-hint">已添加全部</span>
        )}
      </div>
    </div>
  )
}
