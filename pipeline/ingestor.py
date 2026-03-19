import json, pathlib
from rich.console import Console
from rich.table import Table

console = Console()

REQUIRED_INSTRUCT_KEYS = {"instruction", "output"}
REQUIRED_CHAT_KEYS     = {"messages"}

def load_jsonl(path: str) -> list[dict]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    console.print(f"[green]Loaded {len(rows)} rows from {path}[/green]")
    return rows

def validate(rows: list[dict], fmt: str) -> list[dict]:
    required = REQUIRED_INSTRUCT_KEYS if fmt == "instruct" else REQUIRED_CHAT_KEYS
    valid, bad = [], []
    for i, r in enumerate(rows):
        if required.issubset(r.keys()):
            valid.append(r)
        else:
            bad.append(i)
    _print_report(len(rows), len(valid), bad)
    return valid

def _print_report(total, valid, bad_indices):
    t = Table(title="Dataset Validation")
    t.add_column("Metric"); t.add_column("Value", style="cyan")
    t.add_row("Total rows",   str(total))
    t.add_row("Valid rows",   str(valid))
    t.add_row("Dropped rows", str(len(bad_indices)))
    console.print(t)
    if bad_indices:
        console.print(f"[yellow]Dropped row indices: {bad_indices[:10]}{'...' if len(bad_indices)>10 else ''}[/yellow]")
