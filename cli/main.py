import typer, uuid
from rich.console import Console
from rich.panel import Panel

app     = typer.Typer(help="FineTuneKit — local fine-tuning pipeline")
console = Console()

@app.command()
def init(name: str = typer.Argument("my-run")):
    """Scaffold a new fine-tuning project."""
    import pathlib, shutil
    pathlib.Path(f"{name}/data").mkdir(parents=True, exist_ok=True)
    shutil.copy("configs/default.yaml", f"{name}/config.yaml")
    console.print(Panel(f"[green]Project '{name}' created.[/green]\nEdit {name}/config.yaml then run:\n  python cli/main.py run {name}/config.yaml"))

@app.command()
def run(config: str = typer.Argument("configs/default.yaml")):
    """Run the full pipeline."""
    from pipeline.config       import load_config
    from pipeline.ingestor     import load_jsonl, validate
    from pipeline.preprocessor import format_rows, split
    from pipeline.trainer      import train
    from pipeline.evaluator    import evaluate
    from pipeline.exporter     import export
    from registry.store        import register

    run_id = uuid.uuid4().hex[:8]
    cfg    = load_config(config)
    console.print(Panel(f"[bold]FineTuneKit[/bold]  run: {run_id}\nModel: {cfg.model.name}\nDataset: {cfg.dataset.path}"))

    rows        = load_jsonl(cfg.dataset.path)
    rows        = validate(rows, cfg.dataset.format)
    texts       = format_rows(rows, cfg.dataset.format)
    train_texts, val_texts = split(texts, cfg.dataset.train_split)

    metrics     = train(train_texts, val_texts, cfg)
    eval_results= evaluate(val_texts, f"{cfg.training.output_dir}/metrics.json", cfg)
    metrics.update(eval_results)

    record      = export(metrics, cfg, run_id)
    register(record)
    console.print(Panel(f"[bold green]Run {run_id} complete ✓[/bold green]"))

@app.command()
def validate_data(path: str, fmt: str = "instruct"):
    """Validate a dataset file."""
    from pipeline.ingestor import load_jsonl, validate
    validate(load_jsonl(path), fmt)

@app.command()
def runs():
    """List all runs in the registry."""
    from registry.store import list_runs
    list_runs()

if __name__ == "__main__":
    app()
