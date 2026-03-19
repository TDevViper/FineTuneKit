# FineTuneKit 🍎

An open-source local fine-tuning pipeline for LLMs, built for Apple Silicon using MLX.

## Features
- ✅ MLX LoRA training on Apple Silicon (Metal GPU)
- ✅ Config-driven via `config.yaml` + Pydantic validation
- ✅ JSONL dataset ingestion with validation & stats
- ✅ Instruct / Chat / Completion prompt templates
- ✅ ROUGE + loss evaluation
- ✅ SQLite run registry — tracks every experiment
- ✅ CLI: `ftk run`, `ftk runs`, `ftk validate-data`, `ftk init`

## Quickstart
```bash
git clone https://github.com/TDevViper/FineTuneKit.git
cd FineTuneKit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cli/main.py run configs/test.yaml
```

## Project Structure
```
finetunekit/
├── cli/              # CLI entrypoint (ftk commands)
├── pipeline/         # Ingestor, preprocessor, trainer, evaluator, exporter
├── registry/         # SQLite-backed run registry
├── configs/          # YAML config files
└── data/             # Your training data (.jsonl)
```

## Config
```yaml
model:
  name: "mlx-community/Qwen1.5-0.5B-Chat"
  max_tokens: 512

dataset:
  path: "data/train.jsonl"
  format: "instruct"   # instruct | chat | completion
  train_split: 0.8

training:
  epochs: 3
  batch_size: 2
  learning_rate: 2e-5
  lora_rank: 2
  lora_alpha: 2
  lora_dropout: 0.0
  save_every: 10
  output_dir: "runs/"
```

## Dataset Format
```jsonl
{"instruction": "What is gravity?", "output": "Gravity is a force that pulls objects toward each other."}
```

## Week 3 Roadmap
- [ ] FastAPI backend
- [ ] React UI with live loss chart
- [ ] WebSocket streaming training logs

---
Built with ❤️ on Apple Silicon
