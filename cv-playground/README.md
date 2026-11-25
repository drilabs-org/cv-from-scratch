# CV Playground

A shared web app for demoing/learning Computer Vision tasks along the roadmap.
Full spec: `contexts/01-project-summary.md`.

## Architecture

- `backend/` - FastAPI + task registry. Each "task" is a function registered
  via `@register_task` in `backend/app/tasks/`, with metadata (params, docs)
  so the frontend can generate its own UI.
- `frontend/` - React + Vite + shadcn/ui + Tailwind. The UI is generated from
  `GET /api/tasks` and calls `POST /api/tasks/{id}` with `multipart/form-data`
  (image file + `params` JSON).
- CV logic lives in `backend/app/cvlib/` (`imgtool.py`, `mnist.py`,
  `train.py`, `models/`); tasks are only thin wrappers importing it.

## Run

One command for both (backend :8000 + frontend :5173):

```bash
./dev.sh
```

Or run them separately.

Backend (port 8000):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd backend && ../.venv/bin/uvicorn app.main:app --reload --port 8000
```

Frontend, dev mode (port 5173, proxies `/api` -> 8000):

```bash
cd frontend && npm install && npm run dev
``` 

Or build it and let the backend serve it at `/`:

```bash
cd frontend && npm run build
# open http://localhost:8000
```

Auto-generated API docs: http://localhost:8000/docs

## Adding a new task

Add a file `backend/app/tasks/pX_*.py` that registers the task with
`@register_task`, then import it in `backend/app/main.py`. No frontend
changes needed.

Params without a `group` key render as the main always-visible controls;
params sharing a `group` name are collapsed under that heading inside an
"Advanced" disclosure (name groups after pipeline stages, e.g.
`Pre-blur`, `Threshold`).

Docs: point `docs` at a `.md` file under `backend/app/tasks/docs/` (English
by default). Adding a sibling `<name>.<lang>.md` file (e.g.
`p1-blur.vi.md`) automatically adds a language tab in the docs panel. The
markdown renderer supports GFM, mermaid code blocks, and LaTeX math
(`$..$` inline, `$$..$$` display).
