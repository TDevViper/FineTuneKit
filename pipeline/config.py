from pydantic import BaseModel, Field
from typing import Literal
import yaml, pathlib

class ModelConfig(BaseModel):
    name: str
    max_tokens: int = 512

class DatasetConfig(BaseModel):
    path: str
    format: Literal["instruct", "completion", "chat"] = "instruct"
    train_split: float = 0.9

class TrainingConfig(BaseModel):
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    save_every: int = 100
    output_dir: str = "runs/"

class EvalConfig(BaseModel):
    metrics: list[str] = ["loss", "rouge"]

class LoggingConfig(BaseModel):
    level: str = "INFO"
    wandb: bool = False

class FTKConfig(BaseModel):
    model:    ModelConfig
    dataset:  DatasetConfig
    training: TrainingConfig
    eval:     EvalConfig     = Field(default_factory=EvalConfig)
    logging:  LoggingConfig  = Field(default_factory=LoggingConfig)

def load_config(path: str = "configs/default.yaml") -> FTKConfig:
    raw = yaml.safe_load(pathlib.Path(path).read_text())
    return FTKConfig(**raw)
