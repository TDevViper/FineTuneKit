import json, pathlib
from rich.console import Console
from rich.table import Table

console = Console()

def evaluate(val_texts: list[str], metrics_path: str, cfg) -> dict:
    results = {}

    if "rouge" in cfg.eval.metrics:
        results["rouge"] = _rouge(val_texts)

    if "loss" in cfg.eval.metrics:
        p = pathlib.Path(metrics_path)
        if p.exists():
            m = json.loads(p.read_text())
            results["final_loss"] = m.get("final_loss", "n/a")

    _print_results(results)
    return results

def _rouge(texts: list[str]) -> dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        r1, rl = [], []
        for t in texts:
            mid = len(t) // 2
            s = scorer.score(t[:mid], t[mid:])
            r1.append(s["rouge1"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
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
