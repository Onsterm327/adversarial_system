import { useState, useCallback } from 'react'
import { Handle, Position, useReactFlow, type NodeProps } from '@xyflow/react'
import type { NodeData } from '../../types'
import { CATEGORY_META } from '../../types'
import './CardNode.css'

const BACKEND_URL = 'http://127.0.0.1:8765/api/execute'

function toNodeData(data: unknown): NodeData {
  return data as unknown as NodeData
}

interface ProgressInfo {
  message: string
  progress: number
}

/** 后端返回的 summary 结构 */
interface ResultSummary {
  dataset: string
  model: string
  attacks: string[]
  defenses: string[]
  metrics: Record<string, number>  // { "Clean": 95.0, "PGD": 42.3, ... }
  samples: number
}

/** 攻击名称对应的 emoji */
const ATTACK_EMOJI: Record<string, string> = {
  Clean: '🧹',
  PGD: '⚔️',
  FGSM: '💥',
  AutoAttack: '🔥',
}

function getAttackEmoji(name: string): string {
  return ATTACK_EMOJI[name] ?? '⚔️'
}

/** 根据准确率返回颜色 */
function accuracyColor(acc: number): string {
  if (acc >= 80) return '#22c55e'
  if (acc >= 50) return '#f59e0b'
  if (acc >= 25) return '#f97316'
  return '#ef4444'
}

/** 将 summary 作为 JSON 字符串嵌入 event 中传递 */
function parseSummary(raw: unknown): ResultSummary | null {
  if (!raw || typeof raw !== 'object') return null
  const s = raw as Record<string, unknown>
  if (!s.metrics || typeof s.metrics !== 'object') return null
  return {
    dataset: String(s.dataset ?? ''),
    model: String(s.model ?? ''),
    attacks: Array.isArray(s.attacks) ? s.attacks.map(String) : [],
    defenses: Array.isArray(s.defenses) ? s.defenses.map(String) : [],
    metrics: s.metrics as Record<string, number>,
    samples: Number(s.samples ?? 0),
  }
}

export default function ResultNode({ id, data }: NodeProps) {
  const nodeData = toNodeData(data)
  const meta = CATEGORY_META[nodeData.category]
  const { getNodes, getEdges } = useReactFlow()
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<ProgressInfo | null>(null)
  const [summary, setSummary] = useState<ResultSummary | null>(null)
  const [error, setError] = useState<string | null>(null)
  const handleExecute = useCallback(async () => {
    setRunning(true)
    setProgress({ message: '🔍 分析连接链路...', progress: 0 })
    setSummary(null)
    setError(null)

    // Build reverse adjacency: target → [sources] (support multiple parents)
    const allNodes = getNodes()
    const allEdges = getEdges()

    const reverseAdj = new Map<string, string[]>()
    for (const e of allEdges) {
      if (!reverseAdj.has(e.target)) reverseAdj.set(e.target, [])
      reverseAdj.get(e.target)!.push(e.source)
    }

    // BFS backwards from result node, collect all connected upstream nodes
    const collected = new Set<string>()
    const queue: string[] = [id]

    while (queue.length > 0) {
      const nodeId = queue.shift()!
      if (collected.has(nodeId)) continue
      collected.add(nodeId)
      for (const parent of (reverseAdj.get(nodeId) || [])) {
        if (!collected.has(parent)) queue.push(parent)
      }
    }

    // Build pipeline — all collected nodes except the result node itself.
    // Attack nodes expand to their individual selected attacks.
    const pipeline: string[] = []
    for (const nodeId of collected) {
      if (nodeId === id) continue
      const node = allNodes.find(n => n.id === nodeId)
      if (!node) continue
      const nd = toNodeData(node.data)
      if (nd.category === 'attack' && nd.selectedAttacks && nd.selectedAttacks.length > 0) {
        pipeline.push(...nd.selectedAttacks)
      } else {
        pipeline.push(nd.label)
      }
    }

    if (pipeline.length === 0) {
      setRunning(false)
      setError('⚠️ 未连接到任何链路')
      return
    }

    try {
      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chain: pipeline }),
      })

      if (!response.ok) {
        throw new Error(`后端错误: ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('无法读取响应流')
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const raw = line.slice(6)
            try {
              const event = JSON.parse(raw)

              if (event.type === 'step') {
                setProgress({ message: event.message, progress: event.progress })
              } else if (event.type === 'metric_update') {
                // 实时更新：每批次返回当前准确率
                setProgress({ message: event.message, progress: event.progress })
                const totalSamples = Number(event.total_samples ?? 0)
                const liveSummary = parseSummary({
                  metrics: event.metrics,
                  attacks: event.metrics ? Object.keys(event.metrics) : [],
                  dataset: '',
                  model: '',
                  defenses: [],
                  samples: totalSamples,
                })
                if (liveSummary) {
                  // 保留上次的 pipeline 元信息，只更新 metrics 和 samples
                  setSummary(prev => ({
                    ...(prev ?? { dataset: '', model: '', attacks: liveSummary.attacks, defenses: [], samples: totalSamples }),
                    metrics: liveSummary.metrics,
                    attacks: liveSummary.attacks,
                    samples: totalSamples,
                  }))
                }
              } else if (event.type === 'result') {
                setProgress({ message: '✅ 执行完成', progress: 100 })

                // 解析结构化结果（包含完整 pipeline 元信息）
                const parsed = parseSummary(event.summary)
                if (parsed) {
                  setSummary(parsed)
                }
                setRunning(false)
              } else if (event.type === 'error') {
                setError(event.message)
                setRunning(false)
              }
            } catch {
              // skip malformed events
            }
          }
        }
      }
    } catch (err: unknown) {
      setError(`连接后端失败: ${err instanceof Error ? err.message : String(err)}`)
      setRunning(false)
    }
  }, [id, getNodes, getEdges])

  return (
    <div className="card-node" data-category={nodeData.category}>
      <Handle type="target" position={Position.Left} />
      <div className="card-header">
        <span className="card-emoji">{meta.emoji}</span>
        <span className="card-name">{meta.label}</span>
      </div>

      <div className="result-actions">
        <button
          className="result-execute-btn"
          onClick={handleExecute}
          disabled={running}
        >
          {running ? '⏳' : '▶'} 执行
        </button>
      </div>

      {/* Progress bar */}
      {running && progress && (
        <div className="result-progress">
          <div className="progress-bar-track">
            <div
              className="progress-bar-fill"
              style={{ width: `${progress.progress}%` }}
            />
          </div>
          <div className="progress-message">{progress.message}</div>
        </div>
      )}

      {/* Structured result display — shows live during execution and final when done */}
      {summary && (
        <div className="result-metrics-panel">
          {/* Per-attack accuracy cards */}
          <div className="metrics-grid">
            {summary.attacks.map((atkName) => {
              const acc = summary.metrics[atkName]
              const color = acc !== undefined ? accuracyColor(acc) : '#94a3b8'
              return (
                <div
                  className="metric-card"
                  key={atkName}
                  style={{ borderLeftColor: color }}
                >
                  <div className="metric-card-header">
                    <span className="metric-emoji">{getAttackEmoji(atkName)}</span>
                    <span className="metric-name">{atkName}</span>
                  </div>
                  <div className="metric-value" style={{ color }}>
                    {acc !== undefined ? `${acc}%` : '—'}
                  </div>
                  <div className="metric-bar-track">
                    <div
                      className="metric-bar-fill"
                      style={{
                        width: `${acc ?? 0}%`,
                        background: color,
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>

          {/* Footer: sample count */}
          <div className="metrics-footer">
            共 {summary.samples} 个测试样本
          </div>
        </div>
      )}

      {/* Error */}
      {error && !running && (
        <div className="result-output result-error">{error}</div>
      )}

    </div>
  )
}
