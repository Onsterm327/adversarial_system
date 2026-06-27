import { useCallback, useRef, useState, type DragEvent } from 'react'
import {
  ReactFlow,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  Controls,
  Background,
  BackgroundVariant,
  ConnectionMode,
  type Node,
  type Edge,
  type Connection,
  MarkerType,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import type { CardDef, NodeData } from '../types'
import { isValidConnection, getConnectionError } from '../utils/validation'
import DatasetNode from './nodes/DatasetNode'
import ModelNode from './nodes/ModelNode'
import AttackNode from './nodes/AttackNode'
import DefenseNode from './nodes/DefenseNode'
import ResultNode from './nodes/ResultNode'
import ConnectionPanel from './ConnectionPanel'

const nodeTypes = {
  dataset: DatasetNode,
  model: ModelNode,
  attack: AttackNode,
  defense: DefenseNode,
  result: ResultNode,
}

const defaultEdgeOptions = {
  style: { stroke: '#94a3b8', strokeWidth: 2 },
  markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8', width: 14, height: 14 },
  animated: true,
}

let nodeIdCounter = 0
function nextId(): string {
  return `node_${++nodeIdCounter}`
}

function toNodeData(n: Node): NodeData {
  return n.data as unknown as NodeData
}

export default function WorkflowCanvas() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const [showPanel, setShowPanel] = useState(false)
  const { screenToFlowPosition } = useReactFlow()

  const onConnect = useCallback(
    (connection: Connection) => {
      const sourceNode = nodes.find(n => n.id === connection.source)
      const targetNode = nodes.find(n => n.id === connection.target)

      if (sourceNode && targetNode) {
        const srcData = toNodeData(sourceNode)
        const tgtData = toNodeData(targetNode)

        if (!isValidConnection(srcData, tgtData)) {
          const err = getConnectionError(srcData, tgtData)
          alert(err ?? '无效的连接')
          return
        }
      }

      setEdges((eds) => addEdge(connection, eds))
    },
    [nodes, setEdges],
  )

  const isValidConnectionFn = useCallback(
    (connection: Edge | Connection) => {
      const sourceNode = nodes.find(n => n.id === connection.source)
      const targetNode = nodes.find(n => n.id === connection.target)
      if (!sourceNode || !targetNode) return false
      return isValidConnection(toNodeData(sourceNode), toNodeData(targetNode))
    },
    [nodes],
  )

  /** Handle drop from CardPalette */
  const onDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }, [])

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      const raw = e.dataTransfer.getData('application/card')
      if (!raw) return
      const card: CardDef = JSON.parse(raw)

      // Use React Flow's screenToFlowPosition for correct viewport-aware positioning
      const position = screenToFlowPosition({
        x: e.clientX,
        y: e.clientY,
      })

      const newNode: Node = {
        id: nextId(),
        type: card.category,
        position,
        data: {
          cardId: card.id,
          label: card.name,
          category: card.category,
          defenseSubtype: card.defenseSubtype,
        } satisfies NodeData,
      }

      setNodes((nds) => [...nds, newNode])
    },
    [setNodes, screenToFlowPosition],
  )

  return (
    <div
      ref={reactFlowWrapper}
      style={{ flex: 1, height: '100vh', position: 'relative' }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        isValidConnection={isValidConnectionFn}
        onDragOver={onDragOver}
        onDrop={onDrop}
        nodeTypes={nodeTypes}
        defaultEdgeOptions={defaultEdgeOptions}
        connectionMode={ConnectionMode.Loose}
        fitView
        deleteKeyCode={['Backspace', 'Delete']}
      >
        <Controls />
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#e2e8f0" />
      </ReactFlow>

      {/* Connection summary button */}
      <button
        className="conn-panel-trigger"
        onClick={() => setShowPanel(true)}
      >
        📋 查看连接关系
      </button>

      {/* Connection summary panel */}
      {showPanel && (
        <ConnectionPanel
          nodes={nodes}
          edges={edges}
          onClose={() => setShowPanel(false)}
        />
      )}
    </div>
  )
}
