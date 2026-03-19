import json, pathlib, shutil
from datetime import datetime
from rich.console import Console

console = Console()

def export(metrics: dict, cfg, run_id: str):
    out = pathlib.Path(cfg.training.output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id":    run_id,
        "model":     cfg.model.name,
        "dataset":   cfg.dataset.path,
        "created":   datetime.now().isoformat(),
        "metrics":   metrics,
        "adapter":   metrics.get("adapter_path", ""),
    }
    (out / "run.json").write_text(json.dumps(record, indent=2))
    console.print(f"[green]Run exported → {out}/run.json[/green]")
    return record
