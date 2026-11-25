"""Task registry: each "task" is a function registered with metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

TaskHandler = Callable[
    [np.ndarray, dict[str, Any]],
    "tuple[np.ndarray | None, dict[str, Any]]",
]
RuntimeInfoProvider = Callable[[], dict[str, Any] | None]


class TaskExecutionError(Exception):
    """An expected task failure that can be shown to the user."""


@dataclass
class Task:
    id: str
    name: str
    project: str
    description: str
    handler: TaskHandler
    params: list[dict] = field(default_factory=list)
    input: str = "image"
    output: str = "image"  # "image" | "data" | "image+data"
    result_tab: str = "auto"  # "auto" | "input" | "compare" | "output"
    docs: dict[str, str] = field(default_factory=dict)  # lang -> markdown
    docs_path: Path | None = None
    runtime_info: RuntimeInfoProvider | None = None

    def metadata(self) -> dict:
        # re-read the .md files on every request so docs edits need no restart
        if self.docs_path is not None:
            try:
                self.docs = _load_docs(self.docs_path)
            except OSError:
                pass
        return {
            "id": self.id,
            "name": self.name,
            "project": self.project,
            "description": self.description,
            "params": self.params,
            "input": self.input,
            "output": self.output,
            "result_tab": self.result_tab,
            "docs": self.docs,
        }


TASKS: dict[str, Task] = {}


def register_task(
    *,
    id: str,
    name: str,
    project: str,
    description: str,
    params: list[dict] | None = None,
    input: str = "image",
    output: str = "image",
    result_tab: str = "auto",
    docs: str | Path = "",
    runtime_info: RuntimeInfoProvider | None = None,
):
    def deco(fn: TaskHandler) -> TaskHandler:
        docs_path = docs if isinstance(docs, Path) else None
        TASKS[id] = Task(
            id=id,
            name=name,
            project=project,
            description=description,
            handler=fn,
            params=params or [],
            input=input,
            output=output,
            result_tab=result_tab,
            docs=_load_docs(docs),
            docs_path=docs_path,
            runtime_info=runtime_info,
        )
        return fn

    return deco


def _load_docs(docs: str | Path) -> dict[str, str]:
    """docs is a Path -> read the .md file as English plus sibling
    `.<lang>.md` files (`p1-blur.vi.md` adds a "vi" tab); a str -> inline
    English markdown."""
    if isinstance(docs, Path):
        out = {"en": docs.read_text(encoding="utf-8")}
        for lang_file in sorted(docs.parent.glob(f"{docs.stem}.*.md")):
            lang = lang_file.name[: -len(".md")].rsplit(".", 1)[-1]
            out[lang] = lang_file.read_text(encoding="utf-8")
        return out
    return {"en": docs}


def coerce_params(task: Task, raw: dict) -> dict[str, Any]:
    """Apply defaults + validate type/min/max/options per the params metadata."""
    values = {}
    for p in task.params:
        v = raw.get(p["name"], p.get("default"))
        if v is None:
            continue
        if p["type"] == "int":
            try:
                v = int(v)
            except (TypeError, ValueError):
                raise ValueError(f"param '{p['name']}' must be int, got: {v!r}")
        elif p["type"] == "float":
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"param '{p['name']}' must be float, got: {v!r}")
        elif p["type"] == "select":
            if v not in p.get("options", []):
                raise ValueError(f"param '{p['name']}' is invalid: {v!r}")
        if p["type"] in ("int", "float"):
            if "min" in p and v < p["min"]:
                raise ValueError(f"param '{p['name']}' below min {p['min']}: {v}")
            if "max" in p and v > p["max"]:
                raise ValueError(f"param '{p['name']}' above max {p['max']}: {v}")
        values[p["name"]] = v
    return values
