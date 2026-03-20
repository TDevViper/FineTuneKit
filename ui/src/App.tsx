import { useState, useEffect, useRef, useCallback } from "react"
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Legend
} from "recharts"
import axios from "axios"

const API = "http://localhost:8000"

interface LossPoint { iter: number; loss: number }
interface Run {
  run_id: string; model: string; created: string
  metrics: { final_loss: number; rouge?: { rouge1: number; rougeL: number } }
}
interface Model { id: string; label: string; vram: string }
type Nav = "train" | "runs" | "infer" | "export"

const T = {
  bg:       "#0a0a0a",
  surface:  "#111111",
  border:   "#1e1e1e",
  border2:  "#2a2a2a",
  text:     "#ededed",
  muted:    "#666",
  muted2:   "#444",
  accent:   "#5b8df6",
  accentDim:"#1d2d4a",
  green:    "#3dd68c",
  greenDim: "#0d2a1c",
  red:      "#f75c5c",
  yellow:   "#f5a623",
  font:     "\'Berkeley Mono\', \'Fira Code\', monospace",
  fontSans: "\'DM Sans\', \'Inter\', sans-serif",
}

const GLOBAL_CSS = `
  @import url(\'https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap\');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0a0a0a; color: #ededed; font-family: \'DM Sans\', sans-serif; font-size: 13px; line-height: 1.5; -webkit-font-smoothing: antialiased; }
  ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
  @keyframes pulse  { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  @keyframes spin   { to { transform: rotate(360deg); } }
  .fade-in { animation: fadeIn 0.25s ease both; }
  input, select, textarea { background: #111; border: 1px solid #2a2a2a; color: #ededed; border-radius: 6px; padding: 8px 12px; font-size: 13px; font-family: \'DM Sans\', sans-serif; outline: none; transition: border-color 0.15s; }
  input:focus, select:focus, textarea:focus { border-color: #5b8df6; }
  input::placeholder, textarea::placeholder { color: #666; }
  .btn { display: inline-flex; align-items: center; gap: 6px; padding: 7px 16px; border-radius: 6px; border: none; font-size: 13px; font-weight: 500; cursor: pointer; transition: all 0.15s; font-family: \'DM Sans\', sans-serif; }
  .btn:disabled { opacity: 0.35; cursor: not-allowed; }
  .btn-primary { background: #5b8df6; color: #fff; }
  .btn-primary:hover:not(:disabled) { background: #7aaaff; }
  .btn-ghost { background: #1e1e1e; color: #ededed; }
  .btn-ghost:hover:not(:disabled) { background: #2a2a2a; }
  .btn-success { background: #3dd68c; color: #000; }
  .btn-success:hover:not(:disabled) { background: #5cf0ac; }
  .card { background: #111111; border: 1px solid #1e1e1e; border-radius: 10px; padding: 20px; }
  .tag { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
  .tag-green  { background: #0d2a1c; color: #3dd68c; }
  .tag-blue   { background: #1d2d4a; color: #5b8df6; }
  .tag-yellow { background: #2a1f00; color: #f5a623; }
  .tag-muted  { background: #1a1a1a; color: #666; }
`

function Spinner() {
  return <span style={{ display:"inline-block", width:13, height:13, border:"2px solid #2a2a2a", borderTopColor:"#5b8df6", borderRadius:"50%", animation:"spin 0.7s linear infinite" }} />
}

function StatusMsg({ text }: { text: string | null }) {
  if (!text) return null
  const ok = text.startsWith("✅"), skip = text.startsWith("⚠️")
  return <pre style={{ marginTop:12, padding:"10px 14px", borderRadius:6, background: ok?"#0d2a1c":skip?"#1f1800":"#1f0000", color: ok?"#3dd68c":skip?"#f5a623":"#f75c5c", fontSize:12, whiteSpace:"pre-wrap", lineHeight:1.6, border:`1px solid ${ok?"#1a4a30":skip?"#3a2800":"#3a0000"}` }}>{text}</pre>
}

function SectionHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return <div style={{ marginBottom:24 }}><h1 style={{ fontSize:18, fontWeight:600, letterSpacing:"-0.02em", marginBottom:4 }}>{title}</h1>{subtitle && <p style={{ color:"#666", fontSize:13 }}>{subtitle}</p>}</div>
}

function Label({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return <div style={{ fontSize:12, fontWeight:500, color:"#666", marginBottom:6, ...style }}>{children}</div>
}

function Sidebar({ nav, setNav, running }: { nav: Nav; setNav: (n: Nav) => void; running: boolean }) {
  const items: { id: Nav; label: string; icon: string }[] = [
    { id:"train",  label:"Train",     icon:"⚡" },
    { id:"runs",   label:"Runs",      icon:"📊" },
    { id:"infer",  label:"Inference", icon:"🔀" },
    { id:"export", label:"Export",    icon:"📦" },
  ]
  return (
    <aside style={{ width:200, minHeight:"100vh", background:"#111111", borderRight:"1px solid #1e1e1e", display:"flex", flexDirection:"column", padding:"24px 0", flexShrink:0 }}>
      <div style={{ padding:"0 20px 28px" }}>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <div style={{ width:28, height:28, borderRadius:7, background:"linear-gradient(135deg, #5b8df6, #8b5cf6)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:14 }}>⚗</div>
          <div>
            <div style={{ fontWeight:600, fontSize:14, letterSpacing:"-0.02em" }}>FineTuneKit</div>
            <div style={{ fontSize:10, color:"#666", marginTop:1 }}>Apple Silicon LoRA</div>
          </div>
        </div>
      </div>
      <nav style={{ flex:1, padding:"0 10px" }}>
        {items.map(item => {
          const active = nav === item.id
          return (
            <button key={item.id} onClick={() => setNav(item.id)} style={{ width:"100%", display:"flex", alignItems:"center", gap:10, padding:"8px 12px", borderRadius:7, border:"none", cursor:"pointer", background: active?"#1d2d4a":"transparent", color: active?"#5b8df6":"#666", fontWeight: active?500:400, fontSize:13, marginBottom:2, transition:"all 0.15s", fontFamily:"DM Sans, sans-serif" }}>
              <span style={{ fontSize:14 }}>{item.icon}</span>
              {item.label}
              {item.id==="train" && running && <span style={{ marginLeft:"auto" }}><Spinner /></span>}
            </button>
          )
        })}
      </nav>
      <div style={{ padding:"16px 20px 0", borderTop:"1px solid #1e1e1e" }}>
        <div style={{ display:"flex", alignItems:"center", gap:6 }}>
          <div style={{ width:6, height:6, borderRadius:"50%", background: running?"#f5a623":"#3dd68c", ...(running?{animation:"pulse 2s infinite"}:{}) }} />
          <span style={{ fontSize:11, color:"#666" }}>{running?"Training…":"Ready"}</span>
        </div>
      </div>
    </aside>
  )
}

function TrainTab({ models, running, status, lossData, progress, runId, selectedModel, setSelectedModel, dataset, setDataset, uploadInfo, uploading, dragOver, setDragOver, onFileUpload, onStart }: any) {
  const selMeta = models.find((m: Model) => m.id === selectedModel)
  return (
    <div className="fade-in" style={{ maxWidth:760 }}>
      <SectionHeader title="New Training Run" subtitle="Configure and launch a LoRA fine-tuning job" />
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14, marginBottom:16 }}>
        <div className="card">
          <Label>Base Model</Label>
          <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} disabled={running} style={{ width:"100%", marginBottom:8 }}>
            {models.map((m: Model) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
          {selMeta && <div style={{ display:"flex", gap:6 }}><span className="tag tag-blue">{selMeta.vram}</span><span className="tag tag-muted">{selMeta.id.split("/")[1]}</span></div>}
        </div>
        <div className="card">
          <Label>Dataset</Label>
          <div onDrop={e => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if(f) onFileUpload(f) }} onDragOver={e => { e.preventDefault(); setDragOver(true) }} onDragLeave={() => setDragOver(false)} onClick={() => document.getElementById("file-input")?.click()}
            style={{ border:`1px dashed ${dragOver?"#5b8df6":"#2a2a2a"}`, borderRadius:6, padding:"14px 12px", cursor:"pointer", background: dragOver?"#1d2d4a":"transparent", textAlign:"center", transition:"all 0.15s", marginBottom:8 }}>
            {uploading ? <span style={{ color:"#666" }}><Spinner />&nbsp;Uploading…</span> : uploadInfo ? <span style={{ color:"#3dd68c" }}>✓ {uploadInfo.filename}</span> : <span style={{ color:"#666" }}>Drop .jsonl or click to browse</span>}
          </div>
          {uploadInfo && <div style={{ display:"flex", gap:6 }}><span className="tag tag-green">{uploadInfo.valid} valid</span><span className="tag tag-muted">{uploadInfo.total} total</span></div>}
          <input id="file-input" type="file" accept=".jsonl" style={{ display:"none" }} onChange={e => e.target.files?.[0] && onFileUpload(e.target.files[0])} />
        </div>
      </div>
      <div className="card" style={{ marginBottom:20 }}>
        <div style={{ display:"flex", alignItems:"center", gap:14 }}>
          <button className="btn btn-primary" onClick={onStart} disabled={running}>{running ? <><Spinner /> Training…</> : "▶ Start Run"}</button>
          <span style={{ color:"#666", fontSize:12 }}>{status}</span>
          {running && progress.total > 0 && (
            <div style={{ flex:1, maxWidth:220 }}>
              <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                <span style={{ fontSize:11, color:"#666" }}>Progress</span>
                <span style={{ fontSize:11, color:"#666", fontFamily:"monospace" }}>{progress.iter}/{progress.total}</span>
              </div>
              <div style={{ height:3, background:"#2a2a2a", borderRadius:2 }}>
                <div style={{ width:`${(progress.iter/progress.total)*100}%`, height:"100%", background:"#5b8df6", borderRadius:2, transition:"width 0.4s ease" }} />
              </div>
            </div>
          )}
        </div>
      </div>
      {lossData.length > 0 && (
        <div className="card fade-in">
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"baseline", marginBottom:16 }}>
            <Label style={{ margin:0 }}>Live Loss</Label>
            <span style={{ fontSize:11, color:"#666", fontFamily:"monospace" }}>{runId}</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
              <XAxis dataKey="iter" tick={{ fontSize:10, fill:"#666" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize:10, fill:"#666" }} axisLine={false} tickLine={false} width={40} />
              <Tooltip contentStyle={{ background:"#111", border:"1px solid #2a2a2a", borderRadius:6, fontSize:12 }} labelStyle={{ color:"#666" }} formatter={(v: any) => [Number(v).toFixed(4), "loss"]} />
              <Line type="monotone" dataKey="loss" stroke="#5b8df6" dot={false} strokeWidth={1.5} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

function RunsTab({ runs }: { runs: Run[] }) {
  const chartData = runs.slice(0,8).reverse().map(r => ({
    id: r.run_id.slice(0,6),
    loss:   r.metrics?.final_loss ?? 0,
    rouge1: r.metrics?.rouge?.rouge1 ?? 0,
    rougeL: r.metrics?.rouge?.rougeL ?? 0,
  }))
  if (runs.length === 0) return (
    <div className="fade-in" style={{ maxWidth:860 }}>
      <SectionHeader title="Run History" subtitle="0 runs" />
      <div style={{ textAlign:"center", padding:"48px 20px", color:"#666" }}><div style={{ fontSize:32, marginBottom:10 }}>📭</div><div>No runs yet. Start a training run to see results here.</div></div>
    </div>
  )
  const best = [...runs].sort((a,b) => (a.metrics?.final_loss??99)-(b.metrics?.final_loss??99))[0]
  return (
    <div className="fade-in" style={{ maxWidth:860 }}>
      <SectionHeader title="Run History" subtitle={`${runs.length} completed run${runs.length!==1?"s":""}`} />
      {runs.length > 1 && (
        <div className="card" style={{ marginBottom:16 }}>
          <Label style={{ marginBottom:14 }}>Run Comparison</Label>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={chartData} barCategoryGap="30%">
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" vertical={false} />
              <XAxis dataKey="id" tick={{ fontSize:10, fill:"#666" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize:10, fill:"#666" }} axisLine={false} tickLine={false} width={36} />
              <Tooltip contentStyle={{ background:"#111", border:"1px solid #2a2a2a", borderRadius:6, fontSize:12 }} labelStyle={{ color:"#666" }} />
              <Legend wrapperStyle={{ fontSize:11, color:"#666" }} />
              <Bar dataKey="loss"   fill="#f75c5c" radius={[3,3,0,0]} name="Final Loss" />
              <Bar dataKey="rouge1" fill="#5b8df6" radius={[3,3,0,0]} name="ROUGE-1" />
              <Bar dataKey="rougeL" fill="#3dd68c" radius={[3,3,0,0]} name="ROUGE-L" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <div style={{ marginBottom:16, padding:"12px 16px", borderRadius:8, background:"#0d2a1c", border:"1px solid #1a4a30", display:"flex", alignItems:"center", gap:10 }}>
        <span style={{ fontSize:16 }}>🏆</span>
        <div>
          <span style={{ color:"#3dd68c", fontWeight:500 }}>Best run: </span>
          <span style={{ fontFamily:"monospace", color:"#3dd68c", fontSize:12 }}>{best.run_id}</span>
          <span style={{ color:"#666", marginLeft:10, fontSize:12 }}>loss {best.metrics?.final_loss} · ROUGE-1 {best.metrics?.rouge?.rouge1?.toFixed(4)??"—"}</span>
        </div>
      </div>
      <div className="card" style={{ padding:0, overflow:"hidden" }}>
        <table style={{ width:"100%", borderCollapse:"collapse" }}>
          <thead>
            <tr style={{ borderBottom:"1px solid #1e1e1e" }}>
              {["Run ID","Model","Final Loss","ROUGE-1","ROUGE-L","Created"].map(h => (
                <th key={h} style={{ padding:"12px 16px", textAlign:"left", fontSize:11, fontWeight:500, color:"#666", textTransform:"uppercase", letterSpacing:"0.06em" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.map((r,i) => (
              <tr key={r.run_id} style={{ borderBottom: i<runs.length-1?"1px solid #1e1e1e":"none" }}
                onMouseEnter={e=>(e.currentTarget.style.background="#1a1a1a")}
                onMouseLeave={e=>(e.currentTarget.style.background="transparent")}>
                <td style={{ padding:"12px 16px", fontFamily:"monospace", fontSize:12, color:"#5b8df6" }}>{r.run_id}</td>
                <td style={{ padding:"12px 16px", color:"#666", fontSize:12 }}>{r.model?.split("/")[1]??"—"}</td>
                <td style={{ padding:"12px 16px" }}><span className="tag tag-muted" style={{ fontFamily:"monospace" }}>{r.metrics?.final_loss??"—"}</span></td>
                <td style={{ padding:"12px 16px", fontFamily:"monospace", fontSize:12 }}>{r.metrics?.rouge?.rouge1?.toFixed(4)??"—"}</td>
                <td style={{ padding:"12px 16px", fontFamily:"monospace", fontSize:12 }}>{r.metrics?.rouge?.rougeL?.toFixed(4)??"—"}</td>
                <td style={{ padding:"12px 16px", color:"#666", fontSize:11 }}>{r.created?.slice(0,16).replace("T"," ")??"—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function InferTab({ runs }: { runs: Run[] }) {
  const [rid, setRid]       = useState("")
  const [prompt, setPrompt] = useState("")
  const [maxTok, setMaxTok] = useState(200)
  const [busy, setBusy]     = useState(false)
  const [base, setBase]     = useState<any>(null)
  const [ft, setFt]         = useState<any>(null)
  const go = async () => {
    if (!rid || !prompt.trim()) return
    setBusy(true); setBase(null); setFt(null)
    const payload = { prompt, run_id: rid, max_tokens: maxTok }
    try {
      const [br, fr] = await Promise.all([axios.post(`${API}/infer/base`, payload), axios.post(`${API}/infer`, payload)])
      setBase(br.data); setFt(fr.data)
    } catch (e: any) { setBase({ error: e.message }); setFt({ error: e.message }) }
    finally { setBusy(false) }
  }
  const RespBox = ({ data, title, accent }: { data: any; title: string; accent?: string }) => (
    <div className="card" style={{ borderTop: accent?`2px solid ${accent}`:undefined }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12 }}>
        <span style={{ fontWeight:500, color: accent??"#ededed" }}>{title}</span>
        {data?.elapsed!=null && <span className="tag tag-muted" style={{ fontFamily:"monospace" }}>{data.elapsed}s</span>}
      </div>
      {busy && !data ? <div style={{ color:"#666", display:"flex", alignItems:"center", gap:8 }}><Spinner /> Generating…</div>
        : data?.error ? <pre style={{ color:"#f75c5c", fontSize:11, whiteSpace:"pre-wrap" }}>{data.error}</pre>
        : data?.response ? <pre style={{ whiteSpace:"pre-wrap", fontSize:13, lineHeight:1.7, color:"#ededed", fontFamily:"DM Sans, sans-serif" }}>{data.response}</pre>
        : <div style={{ color:"#444" }}>—</div>}
    </div>
  )
  return (
    <div className="fade-in" style={{ maxWidth:900 }}>
      <SectionHeader title="Inference" subtitle="Compare base model vs fine-tuned adapter on the same prompt" />
      <div className="card" style={{ marginBottom:16 }}>
        <div style={{ display:"grid", gridTemplateColumns:"1fr auto", gap:12, marginBottom:12 }}>
          <div>
            <Label>Run</Label>
            <select value={rid} onChange={e => setRid(e.target.value)} style={{ width:"100%" }}>
              <option value="">— select a completed run —</option>
              {runs.map(r => <option key={r.run_id} value={r.run_id}>{r.run_id} · {r.model?.split("/")[1]} · loss {r.metrics?.final_loss}</option>)}
            </select>
          </div>
          <div>
            <Label>Max tokens</Label>
            <input type="number" min={10} max={2000} value={maxTok} onChange={e => setMaxTok(Number(e.target.value))} style={{ width:90 }} />
          </div>
        </div>
        <Label>Prompt</Label>
        <textarea value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="Enter a prompt to test…" style={{ width:"100%", minHeight:80, resize:"vertical", marginBottom:12 }} />
        <button className="btn btn-primary" onClick={go} disabled={busy||!rid||!prompt.trim()}>{busy?<><Spinner /> Generating…</>:"▶ Compare"}</button>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
        <RespBox data={base} title="Base model" />
        <RespBox data={ft}   title="Fine-tuned" accent="#5b8df6" />
      </div>
    </div>
  )
}

function ExportTab({ runs }: { runs: Run[] }) {
  const [rid, setRid]           = useState("")
  const [hfToken, setHfToken]   = useState("")
  const [repoId, setRepoId]     = useState("")
  const [quant, setQuant]       = useState("q4_0")
  const [fuseMsg, setFuseMsg]   = useState<string | null>(null)
  const [ggufMsg, setGgufMsg]   = useState<string | null>(null)
  const [pushMsg, setPushMsg]   = useState<string | null>(null)
  const [fuseLoad, setFuseLoad] = useState(false)
  const [ggufLoad, setGgufLoad] = useState(false)
  const [pushLoad, setPushLoad] = useState(false)
  const [fuseDone, setFuseDone] = useState(false)
  const fuse = async () => {
    if (!rid) return
    setFuseLoad(true); setFuseMsg(null)
    try {
      const r = await axios.post(`${API}/export/fuse`, { run_id: rid })
      if (r.data.error) { setFuseMsg(`❌ ${r.data.error}`) } else { setFuseMsg(`✅ Fused → ${r.data.fused_path}`); setFuseDone(true) }
    } catch (e: any) { setFuseMsg(`❌ ${e.message}`) }
    finally { setFuseLoad(false) }
  }
  const toGGUF = async () => {
    if (!rid) return
    setGgufLoad(true); setGgufMsg(null)
    try {
      const r = await axios.post(`${API}/export/gguf`, { run_id: rid, quantization: quant })
      if (r.data.skipped) setGgufMsg(`⚠️ ${r.data.reason}\n💡 ${r.data.hint}`)
      else if (r.data.error) setGgufMsg(`❌ ${r.data.error}`)
      else setGgufMsg(`✅ GGUF → ${r.data.gguf_path}`)
    } catch (e: any) { setGgufMsg(`❌ ${e.message}`) }
    finally { setGgufLoad(false) }
  }
  const push = async () => {
    if (!rid||!repoId||!hfToken) return
    setPushLoad(true); setPushMsg(null)
    try {
      const r = await axios.post(`${API}/export/push`, { run_id: rid, repo_id: repoId, hf_token: hfToken })
      setPushMsg(r.data.error ? `❌ ${r.data.error}` : `✅ Pushed → ${r.data.url}`)
    } catch (e: any) { setPushMsg(`❌ ${e.message}`) }
    finally { setPushLoad(false) }
  }
  const Step = ({ num, title, subtitle, children }: any) => (
    <div className="card" style={{ marginBottom:14 }}>
      <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:14 }}>
        <div style={{ width:22, height:22, borderRadius:"50%", background:"#1d2d4a", color:"#5b8df6", fontSize:11, fontWeight:600, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>{num}</div>
        <div><div style={{ fontWeight:500 }}>{title}</div>{subtitle&&<div style={{ fontSize:11, color:"#666", marginTop:2 }}>{subtitle}</div>}</div>
      </div>
      {children}
    </div>
  )
  return (
    <div className="fade-in" style={{ maxWidth:680 }}>
      <SectionHeader title="Export" subtitle="Fuse adapters, convert to GGUF, or push to HuggingFace" />
      <div style={{ marginBottom:16 }}>
        <Label>Run</Label>
        <select value={rid} onChange={e => { setRid(e.target.value); setFuseDone(false); setFuseMsg(null) }} style={{ width:360 }}>
          <option value="">— select a run —</option>
          {runs.map(r => <option key={r.run_id} value={r.run_id}>{r.run_id} · {r.model?.split("/")[1]} · loss {r.metrics?.final_loss}</option>)}
        </select>
      </div>
      <Step num="1" title="Fuse adapters into base model" subtitle="Merges LoRA weights permanently. Output saved to runs/<id>/fused/">
        <button className="btn btn-ghost" onClick={fuse} disabled={fuseLoad||!rid}>{fuseLoad?<><Spinner /> Fusing…</>:fuseDone?"✓ Fused — Re-fuse":"Fuse"}</button>
        <StatusMsg text={fuseMsg} />
      </Step>
      <Step num="2" title="Convert to GGUF" subtitle="Optional. Requires llama.cpp for Qwen/Gemma/Phi models.">
        <div style={{ display:"flex", gap:10, alignItems:"center" }}>
          <select value={quant} onChange={e => setQuant(e.target.value)}>{["q4_0","q4_1","q5_0","q5_1","q8_0","f16"].map(q=><option key={q} value={q}>{q}</option>)}</select>
          <button className="btn btn-ghost" onClick={toGGUF} disabled={ggufLoad||!rid}>{ggufLoad?<><Spinner /> Converting…</>:"Convert"}</button>
        </div>
        <StatusMsg text={ggufMsg} />
      </Step>
      <Step num="3" title="Push to HuggingFace Hub" subtitle="Uploads the fused model folder as a private repo.">
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10, marginBottom:12 }}>
          <div><Label style={{ fontSize:11, color:"#666" }}>Repo ID</Label><input value={repoId} onChange={e=>setRepoId(e.target.value)} placeholder="username/model-name" style={{ width:"100%" }} /></div>
          <div><Label style={{ fontSize:11, color:"#666" }}>HF Token</Label><input type="password" value={hfToken} onChange={e=>setHfToken(e.target.value)} placeholder="hf_…" style={{ width:"100%" }} /></div>
        </div>
        <button className="btn btn-success" onClick={push} disabled={pushLoad||!rid||!repoId||!hfToken}>{pushLoad?<><Spinner /> Pushing…</>:"Push to Hub"}</button>
        <StatusMsg text={pushMsg} />
      </Step>
    </div>
  )
}

export default function App() {
  const [nav, setNav]             = useState<Nav>("train")
  const [running, setRunning]     = useState(false)
  const [runId, setRunId]         = useState<string | null>(null)
  const [lossData, setLossData]   = useState<LossPoint[]>([])
  const [status, setStatus]       = useState("idle")
  const [runs, setRuns]           = useState<Run[]>([])
  const [progress, setProgress]   = useState({ iter: 0, total: 0 })
  const [models, setModels]       = useState<Model[]>([])
  const [selModel, setSelModel]   = useState("mlx-community/Qwen1.5-0.5B-Chat")
  const [dataset, setDataset]     = useState("data/train.jsonl")
  const [uploading, setUploading] = useState(false)
  const [uploadInfo, setUploadInfo] = useState<any>(null)
  const [dragOver, setDragOver]   = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const el = document.createElement("style")
    el.textContent = GLOBAL_CSS
    document.head.appendChild(el)
    return () => { document.head.removeChild(el) }
  }, [])

  useEffect(() => {
    fetchRuns()
    axios.get(`${API}/models`).then(r => setModels(r.data.models)).catch(()=>{})
  }, [])

  const fetchRuns = async () => {
    try { const res = await axios.get(`${API}/runs`); setRuns(res.data.runs.slice().reverse()) } catch {}
  }

  const onFileUpload = useCallback(async (file: File) => {
    if (!file.name.endsWith(".jsonl")) { alert("Only .jsonl files supported"); return }
    setUploading(true); setUploadInfo(null)
    const form = new FormData(); form.append("file", file)
    try {
      const res = await axios.post(`${API}/upload`, form)
      if (res.data.error) alert("Upload error: " + res.data.error)
      else { setUploadInfo(res.data); setDataset(res.data.path) }
    } finally { setUploading(false) }
  }, [])

  const onStart = async () => {
    setLossData([]); setRunning(true); setStatus("starting…")
    try {
      const res = await axios.post(`${API}/run`, { config:"configs/test.yaml", model:selModel, dataset })
      if (res.data.error) { setStatus(`error: ${res.data.error}`); setRunning(false); return }
      const id = res.data.run_id; setRunId(id); setStatus("connecting…")
      const ws = new WebSocket(`ws://localhost:8000/ws/logs/${id}`); wsRef.current = ws
      ws.onmessage = (e) => {
        const d = JSON.parse(e.data)
        if (d.iter) { setLossData(prev => [...prev, { iter:d.iter, loss:d.loss }]); setProgress({ iter:d.iter, total:d.total }); setStatus(`iter ${d.iter}/${d.total}  loss ${d.loss}`) }
        else if (d.status) { setStatus(d.status.replace(/_/g," ")) }
        else if (d.done) { setStatus(`done · final loss ${d.final_loss}`); setRunning(false); fetchRuns() }
        else if (d.error) { setStatus(`error: ${d.error}`); setRunning(false) }
      }
      ws.onerror = () => { setStatus("websocket error"); setRunning(false) }
    } catch (e: any) { setStatus(`error: ${e.message}`); setRunning(false) }
  }

  return (
    <div style={{ display:"flex", minHeight:"100vh", background:"#0a0a0a" }}>
      <Sidebar nav={nav} setNav={setNav} running={running} />
      <main style={{ flex:1, padding:"32px 36px", overflowY:"auto" }}>
        {nav==="train"  && <TrainTab models={models} running={running} status={status} lossData={lossData} progress={progress} runId={runId} selectedModel={selModel} setSelectedModel={setSelModel} dataset={dataset} setDataset={setDataset} uploadInfo={uploadInfo} uploading={uploading} dragOver={dragOver} setDragOver={setDragOver} onFileUpload={onFileUpload} onStart={onStart} />}
        {nav==="runs"   && <RunsTab runs={runs} />}
        {nav==="infer"  && <InferTab runs={runs} />}
        {nav==="export" && <ExportTab runs={runs} />}
      </main>
    </div>
  )
}