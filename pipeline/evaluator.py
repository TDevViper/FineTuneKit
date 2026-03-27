import json, pathlib
from rich.console import Console
from rich.table import Table

console = Console()

_ASSISTANT_MARKER = "<|im_start|>assistant\n"

def evaluate(val_texts: list[str], metrics_path: str, cfg) -> dict:
    results = {}
    if "rouge" in cfg.eval.metrics:
        results["rouge"] = _rouge(val_texts, cfg)
    if "loss" in cfg.eval.metrics:
        p = pathlib.Path(metrics_path)
        if p.exists():
            m = json.loads(p.read_text())
            results["final_loss"] = m.get("final_loss", "n/a")
    _print_results(results)
    return results

def _rouge(texts: list[str], cfg) -> dict:
    try:
        from rouge_score import rouge_scorer
        from mlx_lm import load, generate
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        model, tokenizer = load(cfg.model.name)
        r1, rl = [], []
        for t in texts:
            if _ASSISTANT_MARKER in t:
                split_idx = t.index(_ASSISTANT_MARKER) + len(_ASSISTANT_MARKER)
                prompt    = t[:split_idx]
                reference = t[split_idx:].strip()
            else:
                cut       = int(len(t) * 0.8)
                prompt    = t[:cut]
                reference = t[cut:].strip()
            if not reference:
                continue
            prediction = generate(model, tokenizer, prompt=prompt,
                max_tokens=min(256, len(reference.split()) * 2 + 32),
                verbose=False).strip()
            s = scorer.score(reference, prediction)
            r1.append(s["rouge1"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        if not r1:
            return {"error": "No scorable validation examples found"}
        return {"rouge1": round(sum(r1)/len(r1), 4), "rougeL": round(sum(rl)/len(rl), 4)}
    except Exception as e:
        return {"error": str(e)}

def _print_results(results: dict):
    t = Table(title="Evaluation Results")
    t.add_column("Metric"); t.add_column("Value", style="green")
    for k, v in results.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                t.add_row(kk, str(vv))
        else:
            t.add_row(k, str(v))
    console.print(t)
