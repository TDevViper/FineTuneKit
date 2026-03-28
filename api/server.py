import asyncio
import json
import uuid
import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="FineTuneKit API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_log_queues: dict[str, asyncio.Queue] = {}
_active_run: dict = {}
_cancel_flag: dict = {"requested": False}


class RunRequest(BaseModel):
    config: str = "configs/test.yaml"
    model: str = "mlx-community/Qwen1.5-0.5B-Chat"
    dataset: str = "data/train.jsonl"


def _run_training_sync(cfg, run_id: str, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.optimizers import clip_grad_norm
    from mlx_lm import load
    from mlx_lm.tuner import linear_to_lora_layers
    import mlx.optimizers as optim
    from pipeline.mlx_dataset import make_dataset
    from pipeline.ingestor import load_jsonl, validate
    from pipeline.preprocessor import format_rows, split
    from pipeline.evaluator import evaluate
    from pipeline.exporter import export
    from registry.store import register
    import pathlib
    import random as _random

    def emit(msg: dict):
        asyncio.run_coroutine_threadsafe(queue.put(msg), loop)

    try:
        rows = validate(load_jsonl(cfg.dataset.path), cfg.dataset.format)
        texts = format_rows(rows, cfg.dataset.format)
        train_texts, val_texts = split(texts, cfg.dataset.train_split)

        emit({"status": "loading_model", "model": cfg.model.name})
        model, tokenizer = load(cfg.model.name)
        linear_to_lora_layers(model, cfg.training.lora_rank, {
            "rank":    cfg.training.lora_rank,
            "scale":   cfg.training.lora_alpha / cfg.training.lora_rank,
            "dropout": cfg.training.lora_dropout,
        })
        model.train()

        train_data = make_dataset(train_texts, tokenizer, cfg.model.max_tokens)
        val_data   = make_dataset(val_texts,   tokenizer, cfg.model.max_tokens)

        output_dir   = pathlib.Path(cfg.training.output_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = str(output_dir / "adapters.safetensors")

        base_lr   = cfg.training.learning_rate
        optimizer = optim.AdamW(learning_rate=base_lr, weight_decay=0.01)

        def loss_fn(model, tokens, prompt_len):
            x      = mx.array(tokens[:-1])[None]
            y      = mx.array(tokens[1:])
            logits = model(x)[0]
            mask   = (mx.arange(len(y)) >= (prompt_len - 1)).astype(mx.float32)
            n      = mask.sum()
            ce     = nn.losses.cross_entropy(logits, y)
            return (ce * mask).sum() / mx.maximum(n, mx.array(1.0))

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        iters   = cfg.training.epochs * max(len(train_data), 1)
        warmup  = min(10, iters // 4)
        indices = list(range(len(train_data)))
        _random.shuffle(indices)
        idx      = 0
        loss_val = float("nan")

        emit({"status": "training_started", "total_iters": iters})

        for it in range(1, iters + 1):
            lr = base_lr * min(1.0, it / max(warmup, 1))
            optimizer.learning_rate = lr
            if idx >= len(indices):
                _random.shuffle(indices)
                idx = 0
            tokens, prompt_len = train_data[indices[idx]]
            idx += 1
            if len(tokens) < prompt_len + 2:
                continue
            loss, grads  = loss_and_grad(model, tokens, prompt_len)
            grads, gnorm = clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            loss_val = loss.item()
            if _cancel_flag.get("requested"):
                emit({"status": "cancelled", "done": True})
                return
            emit({
                "iter":      it,
                "total":     iters,
                "loss":      round(loss_val, 4),
                "grad_norm": round(gnorm.item(), 3),
                "lr":        round(lr, 8),
            })

        model.save_weights(adapter_file)

        metrics_path = str(output_dir / "metrics.json")
        metrics = evaluate(val_texts, metrics_path, cfg)
        metrics["final_loss"]   = round(loss_val, 4)
        metrics["adapter_path"] = adapter_file

        export(metrics, cfg, run_id)
        register({
            "run_id":  run_id,
            "model":   cfg.model.name,
            "created": datetime.datetime.now().isoformat(),
            "metrics": metrics,
        })

        emit({"done": True, "run_id": run_id, "final_loss": round(loss_val, 4), "metrics": metrics})

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        emit({"error": str(e), "traceback": err, "done": True})
    finally:
        _active_run["running"] = False


@app.post("/run")
async def start_run(req: RunRequest):
    if _active_run.get("running"):
        return {"error": "A run is already in progress", "run_id": _active_run.get("run_id")}

    from pipeline.config import load_config
    try:
        cfg = load_config(req.config)
    except Exception as e:
        return {"error": f"Bad config: {e}"}

    cfg.model.name = req.model
    cfg.dataset.path = req.dataset

    run_id = uuid.uuid4().hex[:8]
    queue: asyncio.Queue = asyncio.Queue()
    _log_queues[run_id] = queue
    _active_run["running"] = True
    _active_run["run_id"]  = run_id
    _cancel_flag["requested"] = False

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_training_sync, cfg, run_id, queue, loop)

    return {"run_id": run_id, "status": "started", "config": req.config}



@app.post("/cancel")
async def cancel_run():
    if not _active_run.get("running"):
        return {"error": "No active run to cancel"}
    _cancel_flag["requested"] = True
    return {"status": "cancel_requested", "run_id": _active_run.get("run_id")}


@app.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    import shutil
    from pathlib import Path as P
    run_dir = P("runs") / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    registry_file = P("runs/registry.json")
    if registry_file.exists():
        import fcntl
        with open(registry_file, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            content = f.read().strip()
            records = [r for r in (json.loads(content) if content else []) if r.get("run_id") != run_id]
            f.seek(0); f.truncate()
            f.write(json.dumps(records, indent=2))
            fcntl.flock(f, fcntl.LOCK_UN)
    return {"deleted": run_id}


@app.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    import shutil
    from pathlib import Path as P
    run_dir = P("runs") / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    registry_file = P("runs/registry.json")
    if registry_file.exists():
        import fcntl
        with open(registry_file, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            content = f.read().strip()
            records = [r for r in (json.loads(content) if content else []) if r.get("run_id") != run_id]
            f.seek(0); f.truncate()
            f.write(json.dumps(records, indent=2))
            fcntl.flock(f, fcntl.LOCK_UN)
    return {"deleted": run_id}


@app.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    import shutil, fcntl
    from pathlib import Path as P
    run_dir = P("runs") / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    registry_file = P("runs/registry.json")
    if registry_file.exists():
        with open(registry_file, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            content = f.read().strip()
            records = [r for r in (json.loads(content) if content else []) if r.get("run_id") != run_id]
            f.seek(0); f.truncate()
            f.write(json.dumps(records, indent=2))
            fcntl.flock(f, fcntl.LOCK_UN)
    return {"deleted": run_id}

@app.get("/runs")
async def list_runs():
    registry_file = Path("runs/registry.json")
    if not registry_file.exists():
        return {"runs": []}
    try:
        records = json.loads(registry_file.read_text())
        return {"runs": records}
    except Exception as e:
        return {"runs": [], "error": str(e)}


@app.websocket("/ws/logs/{run_id}")
async def ws_logs(websocket: WebSocket, run_id: str):
    await websocket.accept()
    queue = _log_queues.get(run_id)
    if not queue:
        await websocket.send_json({"error": f"No active run: {run_id}"})
        await websocket.close()
        return
    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=60.0)
                await websocket.send_json(msg)
                if msg.get("done"):
                    break
            except asyncio.TimeoutError:
                await websocket.send_json({"ping": True})
    except WebSocketDisconnect:
        pass
    finally:
        _log_queues.pop(run_id, None)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_run": _active_run.get("run_id") if _active_run.get("running") else None,
    }


@app.get("/models")
async def list_models():
    return {"models": [
        {"id": "mlx-community/Qwen1.5-0.5B-Chat",       "label": "Qwen 1.5 0.5B Chat",      "vram": "~1GB"},
        {"id": "mlx-community/Qwen1.5-1.8B-Chat",       "label": "Qwen 1.5 1.8B Chat",      "vram": "~2GB"},
        {"id": "mlx-community/Qwen2-0.5B-Instruct-4bit","label": "Qwen2 0.5B Instruct 4bit", "vram": "~0.5GB"},
        {"id": "mlx-community/Qwen2-1.5B-Instruct-4bit","label": "Qwen2 1.5B Instruct 4bit", "vram": "~1GB"},
        {"id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "label": "Mistral 7B Instruct 4bit", "vram": "~4GB"},
        {"id": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit","label": "Llama 3.1 8B Instruct 4bit","vram": "~5GB"},
        {"id": "mlx-community/gemma-2-2b-it-4bit",      "label": "Gemma 2 2B Instruct 4bit", "vram": "~2GB"},
        {"id": "mlx-community/phi-2",                   "label": "Phi-2 2.7B",               "vram": "~2GB"},
    ]}


@app.post("/upload")
async def upload_dataset(file: UploadFile):
    import shutil
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    from pipeline.ingestor import load_jsonl, validate
    try:
        rows = load_jsonl(str(dest))
        valid = validate(rows, "instruct")
        return {"path": str(dest), "total": len(rows), "valid": len(valid), "filename": file.filename}
    except Exception as e:
        return {"error": str(e), "path": str(dest)}



class InferRequest(BaseModel):
    prompt: str
    run_id: str
    max_tokens: int = 200


@app.post("/infer")
async def infer_finetuned(req: InferRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_infer, req.prompt, req.run_id, True, req.max_tokens)
    return result


@app.post("/infer/base")
async def infer_base(req: InferRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_infer, req.prompt, req.run_id, False, req.max_tokens)
    return result


def _run_infer(prompt, run_id, use_adapters, max_tokens):
    import time, json as _json
    from mlx_lm import load, generate
    from pathlib import Path as P

    run_dir      = P("runs") / run_id
    adapter_file = run_dir / "adapters.safetensors"

    model_name = "mlx-community/Qwen1.5-0.5B-Chat"
    meta_path  = run_dir / "run.json"
    if meta_path.exists():
        try:
            model_name = _json.loads(meta_path.read_text()).get("model", model_name)
        except Exception:
            pass

    try:
        if use_adapters and adapter_file.exists():
            model, tokenizer = load(model_name, adapter_path=str(run_dir))
        else:
            model, tokenizer = load(model_name)

        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            nl = chr(10)
            formatted = "<|im_start|>user" + nl + prompt + "<|im_end|>" + nl + "<|im_start|>assistant" + nl
        t0 = time.time()
        response = generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)
        elapsed = round(time.time() - t0, 2)
        return {"response": response, "elapsed": elapsed, "used_adapters": use_adapters}
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# ── Week 6: Export endpoints ─────────────────────────────────────────────────

class FuseRequest(BaseModel):
    run_id: str

class GGUFRequest(BaseModel):
    run_id: str
    quantization: str = "q4_0"

class PushRequest(BaseModel):
    run_id: str
    repo_id: str
    hf_token: str


@app.post("/export/fuse")
async def fuse_adapters(req: FuseRequest):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fuse, req.run_id)

def _fuse(run_id: str) -> dict:
    import subprocess, json as _json
    from pathlib import Path as P
    run_dir   = P("runs") / run_id
    fused_dir = run_dir / "fused"
    model_name = "mlx-community/Qwen1.5-0.5B-Chat"
    meta_path  = run_dir / "run.json"
    if meta_path.exists():
        try:
            model_name = _json.loads(meta_path.read_text()).get("model", model_name)
        except Exception:
            pass
    fused_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["python", "-m", "mlx_lm.fuse",
           "--model", model_name,
           "--adapter-path", str(run_dir),
           "--save-path", str(fused_dir),
           "--dequantize"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            return {"error": proc.stderr[-2000:]}
        return {"fused_path": str(fused_dir), "model": model_name, "run_id": run_id}
    except Exception as e:
        return {"error": str(e)}


@app.post("/export/gguf")
async def convert_gguf(req: GGUFRequest):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _convert_gguf, req.run_id, req.quantization)

def _convert_gguf(run_id: str, quantization: str) -> dict:
    import subprocess
    from pathlib import Path as P
    run_dir   = P("runs") / run_id
    fused_dir = run_dir / "fused"
    gguf_out  = run_dir / f"model-{quantization}.gguf"
    if not fused_dir.exists():
        return {"error": "Fused model not found — run /export/fuse first"}
    # Read model name from run metadata
    import json as _j
    model_name = "mlx-community/Qwen1.5-0.5B-Chat"
    meta = run_dir / "run.json"
    if meta.exists():
        try: model_name = _j.loads(meta.read_text()).get("model", model_name)
        except Exception: pass
    cmd = ["mlx_lm.fuse",
           "--model", model_name,
           "--adapter-path", str(run_dir),
           "--export-gguf",
           "--gguf-path", str(gguf_out)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            return {"error": proc.stderr[-2000:]}
        return {"gguf_path": str(gguf_out), "quantization": quantization, "run_id": run_id}
    except Exception as e:
        return {"error": str(e)}


@app.post("/export/push")
async def push_to_hub(req: PushRequest):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _push, req.run_id, req.repo_id, req.hf_token)

def _push(run_id: str, repo_id: str, hf_token: str) -> dict:
    from pathlib import Path as P
    fused_dir = P("runs") / run_id / "fused"
    if not fused_dir.exists():
        return {"error": "Fused model not found — run /export/fuse first"}
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
        api.upload_folder(folder_path=str(fused_dir), repo_id=repo_id,
                          commit_message=f"Upload fine-tuned model (run {run_id})")
        return {"url": f"https://huggingface.co/{repo_id}", "run_id": run_id}
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        err = err.replace(hf_token, "***") if hf_token in err else err
        return {"error": str(e), "traceback": err[-1000:]}
