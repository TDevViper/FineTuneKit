import { useState, useEffect, useRef, useCallback } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import axios from "axios"

const API = "http://localhost:8000"

interface LossPoint { iter: number; loss: number }
interface Run { run_id: string; model: string; created: string; metrics: { final_loss: number; rouge?: { rouge1: number } } }
interface Model { id: string; label: string; vram: string }

export default function App() {
  const [running, setRunning]         = useState(false)
  const [runId, setRunId]             = useState<string | null>(null)
  const [lossData, setLossData]       = useState<LossPoint[]>([])
  const [status, setStatus]           = useState("idle")
  const [runs, setRuns]               = useState<Run[]>([])
  const [progress, setProgress]       = useState({ iter: 0, total: 0 })
  const [models, setModels]           = useState<Model[]>([])
  const [selectedModel, setSelected]  = useState("mlx-community/Qwen1.5-0.5B-Chat")
  const [dataset, setDataset]         = useState("data/train.jsonl")
  const [uploading, setUploading]     = useState(false)
  const [uploadInfo, setUploadInfo]   = useState<{filename:string; valid:number; total:number} | null>(null)
  const [dragOver, setDragOver]       = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    fetchRuns()
    axios.get(`${API}/models`).then(r => setModels(r.data.models))
  }, [])

  const fetchRuns = async () => {
    const res = await axios.get(`${API}/runs`)
    setRuns(res.data.runs.slice().reverse())
  }

  const uploadFile = useCallback(async (file: File) => {
    if (!file.name.endsWith(".jsonl")) {
      alert("Only .jsonl files supported"); return
    }
    setUploading(true)
    setUploadInfo(null)
    const form = new FormData()
    form.append("file", file)
    try {
      const res = await axios.post(`${API}/upload`, form)
      if (res.data.error) {
        alert("Upload error: " + res.data.error)
      } else {
        setUploadInfo(res.data)
        setDataset(res.data.path)
      }
    } finally {
      setUploading(false)
    }
  }, [])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) uploadFile(file)
  }, [uploadFile])

  const startRun = async () => {
    setLossData([])
    setRunning(true)
    setStatus("starting...")
    const res = await axios.post(`${API}/run`, {
      config: "configs/test.yaml",
      model: selectedModel,
      dataset: dataset
    })
    if (res.data.error) { setStatus(`error: ${res.data.error}`); setRunning(false); return }
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
        setStatus(d.status.replace(/_/g, " "))
      } else if (d.done) {
        setStatus(`done — final loss: ${d.final_loss}`)
        setRunning(false); fetchRuns()
      } else if (d.error) {
        setStatus(`error: ${d.error}`); setRunning(false)
      }
    }
    ws.onerror = () => { setStatus("websocket error"); setRunning(false) }
  }

  const selectedMeta = models.find(m => m.id === selectedModel)

  return (
    <div style={{ fontFamily: "system-ui", maxWidth: 960, margin: "0 auto", padding: "2rem" }}>
      <h1 style={{ fontSize: 24, fontWeight: 600, marginBottom: 4 }}>FineTuneKit</h1>
      <p style={{ color: "#666", marginBottom: 28 }}>Local LoRA fine-tuning on Apple Silicon</p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 24 }}>
        {/* Model selector */}
        <div>
          <label style={{ fontSize: 13, fontWeight: 500, display: "block", marginBottom: 8 }}>Model</label>
          <select
            value={selectedModel} onChange={e => setSelected(e.target.value)} disabled={running}
            style={{ width: "100%", padding: "8px 12px", borderRadius: 8, border: "1px solid #ddd", fontSize: 13 }}
          >
            {models.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
          {selectedMeta && (
            <span style={{ fontSize: 11, color: "#888", marginTop: 4, display: "block" }}>
              {selectedMeta.vram} VRAM · {selectedMeta.id}
            </span>
          )}
        </div>

        {/* Dataset upload */}
        <div>
          <label style={{ fontSize: 13, fontWeight: 500, display: "block", marginBottom: 8 }}>Dataset</label>
          <div
            onDrop={onDrop}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => document.getElementById("file-input")?.click()}
            style={{
              border: `2px dashed ${dragOver ? "#3b82f6" : "#ddd"}`,
              borderRadius: 8, padding: "12px 16px", cursor: "pointer",
              background: dragOver ? "#eff6ff" : "#fafafa",
              transition: "all 0.15s", textAlign: "center"
            }}
          >
            {uploading ? (
              <span style={{ fontSize: 13, color: "#888" }}>Uploading...</span>
            ) : uploadInfo ? (
              <span style={{ fontSize: 13, color: "#22c55e" }}>
                ✓ {uploadInfo.filename} — {uploadInfo.valid}/{uploadInfo.total} valid rows
              </span>
            ) : (
              <span style={{ fontSize: 13, color: "#888" }}>
                Drop .jsonl file or click to browse
              </span>
            )}
          </div>
          <input id="file-input" type="file" accept=".jsonl" style={{ display: "none" }}
            onChange={e => e.target.files?.[0] && uploadFile(e.target.files[0])} />
          <span style={{ fontSize: 11, color: "#aaa", marginTop: 4, display: "block" }}>
            Using: {dataset}
          </span>
        </div>
      </div>

      {/* Start + status */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 28 }}>
        <button onClick={startRun} disabled={running} style={{
          background: running ? "#ccc" : "#000", color: "#fff",
          border: "none", borderRadius: 8, padding: "10px 24px",
          fontSize: 14, cursor: running ? "not-allowed" : "pointer"
        }}>
          {running ? "Training..." : "Start Training Run"}
        </button>
        <span style={{ fontSize: 13, color: "#555" }}>{status}</span>
        {running && progress.total > 0 && (
          <div style={{ flex: 1, background: "#f0f0f0", borderRadius: 999, height: 6, maxWidth: 200 }}>
            <div style={{
              width: `${(progress.iter / progress.total) * 100}%`,
              background: "#3b82f6", height: "100%", borderRadius: 999, transition: "width 0.3s"
            }} />
          </div>
        )}
      </div>

      {/* Live loss chart */}
      {lossData.length > 0 && (
        <div style={{ background: "#f9f9f9", borderRadius: 12, padding: 24, marginBottom: 32 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 16 }}>
            <h2 style={{ fontSize: 15, fontWeight: 500, margin: 0 }}>Live loss</h2>
            <span style={{ fontSize: 12, color: "#888", fontFamily: "monospace" }}>{runId}</span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="iter" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => Number(v).toFixed(4)} />
              <Line type="monotone" dataKey="loss" stroke="#3b82f6" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Run history */}
      <h2 style={{ fontSize: 16, fontWeight: 500, marginBottom: 12 }}>Run history</h2>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #eee", textAlign: "left" }}>
            {["Run ID","Model","Final loss","ROUGE-1","Created"].map(h => (
              <th key={h} style={{ padding: "8px 0", fontWeight: 500 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {runs.map(r => (
            <tr key={r.run_id} style={{ borderBottom: "1px solid #f5f5f5" }}>
              <td style={{ padding: "8px 0", fontFamily: "monospace", color: "#555" }}>{r.run_id}</td>
              <td style={{ padding: "8px 0", color: "#555" }}>{r.model?.split("/")[1] ?? "—"}</td>
              <td style={{ padding: "8px 0" }}>{r.metrics?.final_loss ?? "—"}</td>
              <td style={{ padding: "8px 0" }}>{r.metrics?.rouge?.rouge1?.toFixed(4) ?? "—"}</td>
              <td style={{ padding: "8px 0", color: "#888" }}>{r.created?.slice(0,19).replace("T"," ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
