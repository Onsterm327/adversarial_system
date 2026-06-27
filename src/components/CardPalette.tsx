import { type DragEvent } from 'react'
import { CARDS, getCardsByCategory } from '../data/cards'
import { CATEGORY_META, DEFENSE_SUBTYPE_LABELS } from '../types'
import type { CardDef } from '../types'
import './CardPalette.css'

const SUBTYPE_COLORS: Record<string, string> = {
  input_reconstruction: '#8b5cf6',
  adversarial_training: '#ec4899',
  active_defense:       '#f97316',
  ensemble_defense:     '#14b8a6',
}

function onDragStart(e: DragEvent<HTMLDivElement>, card: CardDef) {
  e.dataTransfer.setData('application/card', JSON.stringify(card))
  e.dataTransfer.effectAllowed = 'copy'
}

function CardItem({ card }: { card: CardDef }) {
  const meta = CATEGORY_META[card.category]
  return (
    <div
      className="palette-card"
      draggable
      onDragStart={(e) => onDragStart(e, card)}
    >
      <span className="pc-emoji">{meta.emoji}</span>
      <span className="pc-name">{card.name}</span>
      {card.defenseSubtype && (
        <span
          className="pc-badge"
          style={{ background: SUBTYPE_COLORS[card.defenseSubtype] ?? '#a855f7' }}
        >
          {DEFENSE_SUBTYPE_LABELS[card.defenseSubtype]}
        </span>
      )}
    </div>
  )
}

export default function CardPalette() {
  const { datasets, models, attacks, defenseGroups, result } = getCardsByCategory()

  return (
    <aside className="card-palette">
      <h1 className="palette-title">🛡️ 对抗防御工作流</h1>
      <p className="palette-subtitle">拖拽卡片到画布中构建实验管线</p>

      <div className="palette-section">
        <div className="palette-section-title">📊 数据集</div>
        {datasets.map(c => <CardItem key={c.id} card={c} />)}
      </div>

      <div className="palette-section">
        <div className="palette-section-title">🧠 模型</div>
        {models.map(c => <CardItem key={c.id} card={c} />)}
      </div>

      <div className="palette-section">
        <div className="palette-section-title">⚔️ 攻击</div>
        {attacks.map(c => <CardItem key={c.id} card={c} />)}
      </div>

      <div className="palette-section">
        <div className="palette-section-title">🛡️ 防御</div>
        {Array.from(defenseGroups.entries()).map(([subtype, cards]) => (
          <div key={subtype} style={{ marginBottom: 6, paddingLeft: 4 }}>
            <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 2 }}>
              {DEFENSE_SUBTYPE_LABELS[subtype as keyof typeof DEFENSE_SUBTYPE_LABELS] ?? subtype}
            </div>
            {cards.map(c => <CardItem key={c.id} card={c} />)}
          </div>
        ))}
      </div>

      <div className="palette-section">
        <div className="palette-section-title">📋 结果</div>
        {result && <CardItem key={result.id} card={result} />}
      </div>
    </aside>
  )
}
