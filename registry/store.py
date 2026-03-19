import json, pathlib
from rich.console import Console
from rich.table import Table

REGISTRY_FILE = pathlib.Path("runs/registry.json")
console = Console()

def _load() -> list:
    if not REGISTRY_FILE.exists():
        return []
    return json.loads(REGISTRY_FILE.read_text())

def _save(records: list):
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(records, indent=2))

def register(record: dict):
    records = _load()
    records.append(record)
    _save(records)
    console.print(f"[green]Registered run: {record['run_id']}[/green]")

def list_runs():
    records = _load()
    t = Table(title="Model Registry")
    t.add_column("Run ID"); t.add_column("Model"); t.add_column("Loss"); t.add_column("Created")
    for r in records:
        loss = str(r.get("metrics", {}).get("final_loss", "n/a"))
        t.add_row(r["run_id"], r["model"], loss, r["created"][:19])
    console.print(t)
    return records
