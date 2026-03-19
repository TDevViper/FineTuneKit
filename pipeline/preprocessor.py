import random
from rich.console import Console

console = Console()

INSTRUCT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

def format_rows(rows: list[dict], fmt: str) -> list[str]:
    if fmt == "instruct":
        return [INSTRUCT_TEMPLATE.format(**r) for r in rows]
    if fmt == "completion":
        return [r["text"] for r in rows]
    if fmt == "chat":
        return [_format_chat(r["messages"]) for r in rows]
    raise ValueError(f"Unknown format: {fmt}")

def _format_chat(messages: list[dict]) -> str:
    out = ""
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "user":
            out += f"<s>[INST] {content} [/INST] "
        else:
            out += f"{content} </s>"
    return out

def split(texts: list[str], train_ratio: float) -> tuple[list, list]:
    random.shuffle(texts)
    n = int(len(texts) * train_ratio)
    train, val = texts[:n], texts[n:]
    console.print(f"[blue]Split → train: {len(train)}  val: {len(val)}[/blue]")
    return train, val
