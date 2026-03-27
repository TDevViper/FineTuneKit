import json, pathlib, fcntl
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
    with open(REGISTRY_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(records, indent=2))
        fcntl.flock(f, fcntl.LOCK_UN)

def register(record: dict):
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        content = f.read().strip()
        records = json.loads(content) if content else []
        records.append(record)
        f.seek(0)
        f.truncate()
        f.write(json.dumps(records, indent=2))
        fcntl.flock(f, fcntl.LOCK_UN)
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
