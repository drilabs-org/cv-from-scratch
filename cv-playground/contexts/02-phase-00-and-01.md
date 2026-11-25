---
title: "Phase 0 + Phase 1: implementation log"
description: "What was actually built for the scaffold and the project-1 tasks, decisions made along the way, and the conventions later phases must follow."
---

# Phase 0 + Phase 1: implementation log

Companion to `01-project-summary.md`. That file is the contract; this one logs how phases 0-1 were actually implemented, including decisions that are not visible in the spec. Later phases should copy these patterns instead of inventing new ones.

## Phase 0: scaffold

### What was built

- Backend `backend/app/`: `main.py` (3 endpoints), `registry.py` (`Task` dataclass, `TASKS` dict, `register_task` decorator, `coerce_params`), `tasks/` package.
- Endpoints: `GET /health`, `GET /api/tasks`, `POST /api/tasks/{task_id}` (multipart `file` + `params` JSON -> `{image: base64 PNG | null, data}`, optionally with separate `runtime` metadata).
- Static mount of `frontend/dist` at `/` only if the directory exists, so the backend runs fine before the first frontend build.
- `coerce_params` validates server-side (defaults, int/float cast, min/max, select options) -> 422 on failure. Handlers never see raw strings.
- Frontend: React 19 + Vite 8 + TypeScript, Tailwind 4 via `@tailwindcss/vite` (CSS-first, no `tailwind.config`), shadcn `base-nova` on `@base-ui/react`, `lucide-react` icons, `oxlint` for lint.
- `dev.sh`: creates `.venv`, installs both dependency sets if missing, runs uvicorn `--reload` + vite, traps Ctrl+C to kill both.
- `requirements.txt` is a full `pip freeze` pin; `opencv-python-headless` chosen since no GUI is needed server-side.

### Decisions made here

- **No standalone demo task**: the spec's throwaway `grayscale` demo merged directly into the real `p1-grayscale`, so no dead code remained after phase 0.
- **Docs moved from inline string to files** early: `docs` accepts a `Path`, loaded as `{"en": ...}` plus sibling `.<lang>.md` files, and re-read on every `GET /api/tasks` so docs edits need no restart.

## Phase 1: project 1 tasks

### Backend

- `backend/app/tasks/p1_opencv.py`: 5 tasks wrapping `app/cvlib/imgtool.py`, imported as a normal package module.
- Registered in `main.py` with `from app.tasks import p1_opencv  # noqa: F401`.
- `p1-count` ended up with 7 params because `count_objects()` already supports them; it sets `result_tab="output"` since an annotated image on the Compare slider is less useful than seeing it large.
- **Params split main vs advanced**: any param with a `group` key renders inside an "Advanced" collapsible (`@base-ui/react` `Collapsible`), one heading per group in declaration order; params without `group` stay always visible. Groups mirror pipeline stages (`Pre-blur`, `Threshold`, `Adaptive`, `Morphology`, `Canny`, `Gaussian`). `p1-count` keeps only `min_area` as main.
- Odd `ksize`/`block` is enforced inside the handler (`even -> +1`) rather than in validation; `step: 2` in metadata already guides the UI.
- `imgtool` signatures grew pass-through knobs with safe defaults: `to_blur(sigma)`, `to_edges(ksize, aperture, l2)`, `to_binary(block, C)`, `count_objects(block, C)`. The CLI still runs standalone since every new arg has a default.
- Docs: 5 file triples under `tasks/docs/` (`<task-id>.md` + `<task-id>.vi.md` + `<task-id>.spec.md`). Main docs are long-form: concept -> place in pipeline (mermaid) -> theory with KaTeX math -> the OpenCV call -> param meanings -> tips -> "try it". The `.spec.md` file is an English demo tuning reference (I/O contract, param table, presets, symptom fixes) rendered as a third "Spec" tab.

### Frontend

- `ResultView`: tab bar **Input / Output / Compare**; Compare is a draggable before/after slider (`CompareView`). Default tab after a run: `result_tab` metadata if set, else `output`. `compare` is only honored when both input and output images exist.
- `ImageViewer`: wheel/button zoom (0.05x-16x), pan, fullscreen, download; `viewerZoom` is lifted to `ResultView` so input and output stay in sync. `DropZone` renders its preview through `ImageViewer`, reusing the same toolbar (`ViewerToolButton`).
- `DropZone`: drag/drop + click-to-browse + example thumbnails. Examples are a hardcoded `EXAMPLES` list served from `frontend/public/examples/` (`sample.png`, `sample2.png`, `sample3.png`), fetched as blobs into `File` objects. Clear button calls `onClear` -> `App.clearImage`, which revokes the object URL.
- `ParamsForm`: `select` -> shadcn `Select`; `int`/`float` -> `ElasticSlider` (custom, motion-based, rubber-band + decile snap; `step` from metadata, default 1 / 0.1). Params carrying `group` go into an "Advanced" `Collapsible` (base-ui), grouped under per-stage headings.
- `TaskSidebar`: groups by `task.project`; icon chosen by regex on task id/name (`ICON_RULES`).
- `DocsPanel`: `react-markdown` + `remark-gfm` + `remark-math` + `rehype-katex` + `rehype-highlight`; fenced `mermaid` blocks rendered via `pele`; docs tabs (EN / Spec / VI) when more than one `.<suffix>.md` file exists, labels via `LANG_LABELS`. `index.css` reverts `list-style` inside `.markdown-body` because Tailwind preflight strips bullets.
- `ThemeToggle`: toggles `.dark` on `documentElement`, persisted in `localStorage['theme']`; `index.html` has an inline script applying the saved theme before React mounts (no flash).
- Cmd/Ctrl+Enter runs the task; an elapsed-ms badge shows after each run.

## Conventions for later phases

1. **Adding a task** = new `backend/app/tasks/p<N>_*.py` + one import line in `main.py`. Nothing else on the backend.
2. **Task ids** are `p<N>-<slug>`; `project` is a short English tag (`opencv-basics`, `mnist-digits`, ...) because the sidebar groups by it.
3. **Handlers stay thin**: they receive a BGR `ndarray` and coerced params, return `(img | None, flat data dict)`. No file upload handling, no I/O except lazy model loading. Real CV logic lives in `app/cvlib/`; when a task needs a new knob, add it to the `cvlib` function as a defaulted kwarg (see `to_blur(sigma)`, `to_edges(ksize, aperture, l2)`, `to_binary(block, C)`) so the standalone scripts keep working.
4. **Params metadata is the contract**: always declare `type`, `min`, `max`, `step`, `default`, `options`, `label`, and `group` where relevant; both the generated form and `coerce_params` consume it. Keep 1-3 primary controls ungrouped (always visible); put stage-level knobs into `group`s named after the pipeline stage, rendered under the "Advanced" disclosure in declaration order. Params OpenCV requires odd (`ksize`, `block`) use `step: 2` in metadata plus `even -> +1` inside the handler. Never trust raw values inside a handler.
5. **`data` shape**: flat dict, short keys. Scalars render as badges; arrays of objects render as `key: value` rows. For classification tasks use `[{label, prob}, ...]` so one confidence-bar renderer serves p2 and p3 alike.
6. **`result_tab`**: set it explicitly when `auto` is wrong. Annotated/dual outputs -> `output`; tasks where the input itself is the interesting view -> `input`; side-by-side comparisons -> `compare`.
7. **Docs**: write the `<task-id>.md` + `<task-id>.vi.md` + `<task-id>.spec.md` triple in the same change as the task, following the p1 structure (concept -> pipeline -> theory/math -> API call -> params -> tips -> try it). Renderer supports GFM, `mermaid` fences, `$..$`/`$$..$$` KaTeX.
8. **Model weights** live under `app/cvlib/models/` (e.g. `mnist_cnn.pt`). Load lazily on first task call and cache at module level; never at import time, which would slow backend startup and every task list.
9. **Frontend stays metadata-driven**: new UI needs (confidence bars, box overlays) become generic renderers keyed on metadata or data shape, never `if (task.id === 'p2-digit')`.
10. **Verification loop used so far**: `./dev.sh` up, `curl /api/tasks`, `curl -F file=@img -F params='{}' /api/tasks/<id>`, then screenshot/interact via `playwright-cli` (logs in `.playwright-cli/`); `npm run lint` (oxlint) + `tsc -b && vite build` before calling a phase done.
