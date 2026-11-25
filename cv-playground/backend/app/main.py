import base64
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from app.registry import TASKS, TaskExecutionError, coerce_params
from app.tasks import p1_opencv  # noqa: F401  # imported to register tasks
from app.tasks import p2_mnist  # noqa: F401  # imported to register tasks

app = FastAPI(title="CV Playground")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/tasks")
def list_tasks() -> list[dict]:
    return [t.metadata() for t in TASKS.values()]


@app.post("/api/tasks/{task_id}")
async def run_task(
    task_id: str,
    file: UploadFile = File(...),
    params: str = Form("{}"),
) -> dict:
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, f"task not found: {task_id}")

    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"file cannot be decoded as an image: {file.filename}")

    try:
        values = coerce_params(task, json.loads(params))
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(422, f"invalid params: {e}")

    try:
        out_img, data = task.handler(img, values)
    except TaskExecutionError as e:
        raise HTTPException(500, str(e)) from e

    encoded = None
    if out_img is not None:
        ok, buf = cv2.imencode(".png", out_img)
        if not ok:
            raise HTTPException(500, "failed to encode output image")
        encoded = base64.b64encode(buf.tobytes()).decode()
    result = {"image": encoded, "data": data}
    if task.runtime_info is not None:
        runtime_info = task.runtime_info()
        if runtime_info is not None:
            result["runtime"] = runtime_info
    return result


FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount(
        "/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend"
    )
