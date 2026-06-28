import { useCallback, useState } from 'react'
import { Handle, Position, useReactFlow, useStore, type NodeProps } from '@xyflow/react'
import type { NodeData } from '../../types'
import { CATEGORY_META } from '../../types'
import './CardNode.css'

const DEFAULT_ATTACKS = ['Clean']

export default function AttackNode({ id, data }: NodeProps) {
  const nodeData = data as unknown as NodeData
  const meta = CATEGORY_META[nodeData.category]
  const attacks: string[] = nodeData.selectedAttacks ?? DEFAULT_ATTACKS
  const availableAttacks: string[] = nodeData.availableAttacks ?? []
  const [menuOpen, setMenuOpen] = useState(false)
  const { setNodes } = useReactFlow()
  const allNodes = useStore(s => s.nodes)

  const updateAttacks = useCallback(
    (newAttacks: string[]) => {
      setNodes(
        allNodes.map(n =>
          n.id === id
            ? { ...n, data: { ...n.data, selectedAttacks: newAttacks } }
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
      if (name === 'Clean') return
      updateAttacks(attacks.filter(a => a !== name))
    },
    [attacks, updateAttacks],
  )

  const availableToAdd = availableAttacks.filter(a => !attacks.includes(a))

  return (
    <div className="card-node attack-node" data-category={nodeData.category}>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />

      <div className="attack-header">
        <span className="attack-header-emoji">{meta.emoji}</span>
        <span className="attack-header-name">{nodeData.label}</span>
      </div>

      {/* Attack list */}
      <div className="card-tags">
        {attacks.map(name => (
          <div className={`card-tag tag-attack tag-${name}`} key={name}>
            <span>{name}</span>
            {name !== 'Clean' && (
              <button
                className="tag-remove"
                onClick={() => removeAttack(name)}
                title={`移除 ${name}`}
              >
                ×
              </button>
            )}
          </div>
        ))}
      </div>

      {/* Add attack button */}
      {availableToAdd.length > 0 && (
        <div className="attack-add-row">
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
        </div>
      )}
    </div>
  )
}
