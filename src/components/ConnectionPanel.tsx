import type { Node, Edge } from '@xyflow/react'
import type { NodeData } from '../types'
import { CATEGORY_META, DEFENSE_SUBTYPE_LABELS } from '../types'
import './ConnectionPanel.css'

interface Props {
  nodes: Node[]
  edges: Edge[]
  onClose: () => void
}

function toNodeData(n: Node): NodeData {
  return n.data as unknown as NodeData
}

/** Build readable connection chains from edges */
function buildChains(nodes: Node[], edges: Edge[]): string[][] {
  if (edges.length === 0) return []

  // Build adjacency map
  const children = new Map<string, string[]>()
  const parents = new Map<string, string[]>()
  for (const e of edges) {
    if (!children.has(e.source)) children.set(e.source, [])
    children.get(e.source)!.push(e.target)
    if (!parents.has(e.target)) parents.set(e.target, [])
    parents.get(e.target)!.push(e.source)
  }

  // Find root nodes (no incoming edges)
  const roots = nodes.filter(n => !parents.has(n.id)).map(n => n.id)

  const chains: string[][] = []

  function dfs(nodeId: string, path: string[]) {
    const currentPath = [...path, nodeId]
    const kids = children.get(nodeId)
    if (!kids || kids.length === 0) {
      chains.push(currentPath)
    } else {
      for (const kid of kids) {
        dfs(kid, currentPath)
      }
    }
  }

  // Start DFS from each root
  if (roots.length > 0) {
    for (const root of roots) {
      dfs(root, [])
    }
  } else {
    // If no clear root (cycles or all have parents), start from any
    const visited = new Set<string>()
    for (const n of nodes) {
      if (!visited.has(n.id) && !parents.has(n.id)) {
        dfs(n.id, [])
        visited.add(n.id)
      }
    }
  }

  return chains
}

function getNodeColor(data: NodeData): string {
  if (data.category === 'defense' && data.defenseSubtype) {
    const colors: Record<string, string> = {
      input_reconstruction: '#8b5cf6',
      adversarial_training: '#ec4899',
      active_defense: '#f97316',
      ensemble_defense: '#14b8a6',
    }
    return colors[data.defenseSubtype] ?? CATEGORY_META.defense.color
  }
  return CATEGORY_META[data.category]?.color ?? '#94a3b8'
}

export default function ConnectionPanel({ nodes, edges, onClose }: Props) {
  const chains = buildChains(nodes, edges)
  const nodeMap = new Map(nodes.map(n => [n.id, n]))

  // Find nodes not in any chain
  const chainedIds = new Set(chains.flat())
  const pendingNodes = nodes.filter(n => !chainedIds.has(n.id))

  return (
    <div className="conn-panel-overlay" onClick={onClose}>
      <div className="conn-panel-card" onClick={e => e.stopPropagation()}>
        <div className="conn-panel-header">
          <h2>📋 当前连接关系</h2>
          <button className="conn-panel-close" onClick={onClose}>✕</button>
        </div>

        <div className="conn-panel-body">
          {chains.length === 0 && pendingNodes.length === 0 && (
            <p className="empty-msg">画布中暂无节点，请从左侧拖拽卡片到画布。</p>
          )}

          {chains.length > 0 && (
            <>
              <div className="section-title">🔗 连接链路</div>
              {chains.map((chain, i) => (
                <div className="conn-chain" key={i}>
                  {chain.map((nodeId, j) => {
                    const node = nodeMap.get(nodeId)
                    const data = node ? toNodeData(node) : null
                    return (
                      <span className="chain-segment" key={nodeId}>
                        {j > 0 && <span className="chain-arrow">→</span>}
                        <span
                          className="chain-node"
                          style={{ background: data ? getNodeColor(data) : '#94a3b8' }}
                        >
                          {data?.label ?? nodeId}
                        </span>
                      </span>
                    )
                  })}
                </div>
              ))}
            </>
          )}

          {pendingNodes.length > 0 && (
            <>
              <div className="section-title">📌 未连接节点</div>
              <div className="pending-tags">
                {pendingNodes.map(n => {
                  const data = toNodeData(n)
                  return (
                    <span
                      key={n.id}
                      className="pending-tag"
                      style={{ background: getNodeColor(data) }}
                    >
                      {data.label}
                    </span>
                  )
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
