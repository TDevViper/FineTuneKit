from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class ProcessState(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"

@dataclass
class TrainRun:
    run_id:     str
    model:      str
    dataset:    str
    state:      ProcessState = ProcessState.PENDING
    loss:       list[float]  = field(default_factory=list)
    metrics:    dict         = field(default_factory=dict)
    error:      Optional[str] = None
