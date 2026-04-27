from __future__ import annotations

import json
from pathlib import Path

from calc_flow.pipeline import Pipeline


class Checkpoint:
    """A snapshot of pipeline state at a point in time."""

    def __init__(self, pipeline_name: str, offset: int, state: dict) -> None:
        self.pipeline_name = pipeline_name
        self.offset = offset
        self.state = state

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline_name,
            "offset": self.offset,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Checkpoint:
        return cls(data["pipeline"], data["offset"], data["state"])


class CheckpointManager:
    """Manages checkpoint persistence and recovery for fault tolerance."""

    def __init__(self, directory: str | Path = ".calc-flow-checkpoints") -> None:
        self._dir = Path(directory)

    def save(self, pipeline: Pipeline, batch_offset: int) -> Checkpoint:
        state = pipeline.snapshot()
        cp = Checkpoint(pipeline.name, batch_offset, state)
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(pipeline.name)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(json.dumps(cp.to_dict()) + "\n", encoding="utf-8")
        tmp_path.replace(path)
        return cp

    def load(self, pipeline_name: str) -> Checkpoint | None:
        path = self._path_for(pipeline_name)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Checkpoint.from_dict(data)

    def recover(self, pipeline: Pipeline) -> int:
        """Restore pipeline state and return the offset to resume from."""
        cp = self.load(pipeline.name)
        if cp is None:
            return 0
        pipeline.restore(cp.state)
        return cp.offset

    def clear(self, pipeline_name: str) -> None:
        self._path_for(pipeline_name).unlink(missing_ok=True)

    def _path_for(self, pipeline_name: str) -> Path:
        return self._dir / f"{pipeline_name}.json"
