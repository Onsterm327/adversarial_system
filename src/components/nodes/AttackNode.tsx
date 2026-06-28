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
      if (name === 'Clean') return
      updateAttacks(attacks.filter(a => a !== name))
    },
    [attacks, updateAttacks],
  )

  const availableToAdd = AVAILABLE_ATTACKS.filter(a => !attacks.includes(a))

  return (
    <div className="card-node attack-node" data-category={nodeData.category}>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />

      <div className="attack-header">
        <span className="attack-header-emoji">{meta.emoji}</span>
        <span className="attack-header-name">测试 ({attacks.length})</span>
      </div>

      {/* Attack list */}
      <div className="attack-list">
        {attacks.map(name => (
          <div className={`attack-tag ${name}`} key={name}>
            <span className="attack-tag-name">{name}</span>
            {name !== 'Clean' ? (
              <button
                className="attack-tag-remove"
                onClick={() => removeAttack(name)}
                title={`移除 ${name}`}
              >
                ×
              </button>
            ) : (
              <span className="attack-tag-spacer" />
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
