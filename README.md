# FineTuneKit

Local LoRA fine-tuning on Apple Silicon — no cloud, no cost.

![Dark UI](ui/public/screenshot.png)

## Features
- **Train** — LoRA fine-tuning via MLX with live loss chart
- **Runs** — Compare runs with bar charts, ROUGE metrics, best run highlight
- **Inference** — Side-by-side base vs fine-tuned comparison
- **Export** — Fuse adapters, convert to GGUF, push to HuggingFace Hub

## Requirements
- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- Node.js 18+

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install mlx mlx-lm fastapi uvicorn huggingface_hub rouge-score rich pyyaml
cd ui && npm install && npm run build && cd ..
uvicorn api.server:app --port 8000
cd ui && npx serve -s build
```

## Usage
Open `http://localhost:3000` — upload a `.jsonl` dataset, pick a model, hit **Start Run**.
