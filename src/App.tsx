import { ReactFlowProvider } from '@xyflow/react'
import CardPalette from './components/CardPalette'
import WorkflowCanvas from './components/WorkflowCanvas'
import './App.css'

export default function App() {
  return (
    <div className="app-layout">
      <ReactFlowProvider>
        <CardPalette />
        <WorkflowCanvas />
      </ReactFlowProvider>
    </div>
  )
}
