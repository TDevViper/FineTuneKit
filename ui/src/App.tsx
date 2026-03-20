import { useState, useEffect, useRef } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import axios from "axios"

const API = "http://localhost:8000"

interface LossPoint { iter: number; loss: number }
interface Run { run_id: string; model: string; created: string; metrics: { final_loss: number; rouge?: { rouge1: number } } }

export default function App() {
  const [running, setRunning]     = useState(false)
  const [runId, setRunId]         = useState<string | null>(null)
  const [lossData, setLossData]   = useState<LossPoint[]>([])
  const [status, setStatus]       = useState("idle")
  const [runs, setRuns]           = useState<Run[]>([])
  const [progress, setProgress]   = useState({ iter: 0, total: 0 })
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => { fetchRuns() }, [])

  const fetchRuns = async () => {
    const res = await axios.get(`${API}/runs`)
    setRuns(res.data.runs.reverse())
  }

  const startRun = async () => {
    setLossData([])
    setRunning(true)
    setStatus("starting...")
    const res = await axios.post(`${API}/run`, { config: "configs/test.yaml" })
    const id = res.data.run_id
    setRunId(id)
    setStatus("connecting...")

    const ws = new WebSocket(`ws://localhost:8000/ws/logs/${id}`)
    wsRef.current = ws

    ws.onmessage = (e) => {
      const d = JSON.parse(e.data)
      if (d.iter) {
        setLossData(prev => [...prev, { iter: d.iter, loss: d.loss }])
        setProgress({ iter: d.iter, total: d.total })
        setStatus(`training — iter ${d.iter}/${d.total}`)
      } else if (d.status) {
        setStatus(d.status.replace("_", " "))
      } else if (d.done) {
        setStatus(`done — final loss: ${d.final_loss}`)
        setRunning(false)
        fetchRuns()
      } else if (d.error) {
        setStatus(`error: ${d.error}`)
        setRunning(false)
      }
    }
    ws.onerror = () => { setStatus("websocket error"); setRunning(false) }
  }

  return (
    <div style={{ fontFamily: "system-ui", maxWidth: 900, margin: "0 auto", padding: "2rem" }}>
      <h1 style={{ fontSize: 24, fontWeight: 600, marginBottom: 4 }}>FineTuneKit</h1>
      <p style={{ color: "#666", marginBottom: 24 }}>Local LoRA fine-tuning on Apple Silicon</p>

      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 24 }}>
        <button
          onClick={startRun}
          disabled={running}
          style={{
            background: running ? "#ccc" : "#000", color: "#fff",
            border: "none", borderRadius: 8, padding: "10px 24px",
            fontSize: 14, cursor: running ? "not-allowed" : "pointer"
          }}
        >
          {running ? "Training..." : "Start Training Run"}
        </button>
        <span style={{ fontSize: 13, color: "#555" }}>{status}</span>
        {running && (
          <span style={{ fontSize: 13, color: "#888" }}>
            {progress.iter}/{progress.total} iters
          </span>
        )}
      </div>

      {lossData.length > 0 && (
        <div style={{ background: "#f9f9f9", borderRadius: 12, padding: 24, marginBottom: 32 }}>
          <h2 style={{ fontSize: 15, fontWeight: 500, marginBottom: 16 }}>Live loss — run {runId}</h2>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="iter" tick={{ fontSize: 11 }} label={{ value: "iter", position: "insideBottom", offset: -2, fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => Number(v).toFixed(4)} />
              <Line type="monotone" dataKey="loss" stroke="#3b82f6" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <h2 style={{ fontSize: 16, fontWeight: 500, marginBottom: 12 }}>Run history</h2>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #eee", textAlign: "left" }}>
            <th style={{ padding: "8px 0", fontWeight: 500 }}>Run ID</th>
            <th style={{ padding: "8px 0", fontWeight: 500 }}>Model</th>
            <th style={{ padding: "8px 0", fontWeight: 500 }}>Final loss</th>
            <th style={{ padding: "8px 0", fontWeight: 500 }}>ROUGE-1</th>
            <th style={{ padding: "8px 0", fontWeight: 500 }}>Created</th>
          </tr>
        </thead>
        <tbody>
          {runs.map(r => (
            <tr key={r.run_id} style={{ borderBottom: "1px solid #f0f0f0" }}>
              <td style={{ padding: "8px 0", fontFamily: "monospace", color: "#555" }}>{r.run_id}</td>
              <td style={{ padding: "8px 0", color: "#555" }}>{r.model.split("/")[1]}</td>
              <td style={{ padding: "8px 0" }}>{r.metrics?.final_loss ?? "—"}</td>
              <td style={{ padding: "8px 0" }}>{r.metrics?.rouge?.rouge1?.toFixed(4) ?? "—"}</td>
              <td style={{ padding: "8px 0", color: "#888" }}>{r.created?.slice(0, 19).replace("T", " ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
