---
title: "Phase 2: MNIST digit recognition - implementation context"
description: "Everything needed to implement phase 2 in one place: current state, the exact model and preprocessing pipeline from the project docs, ordered work items, and decisions to make."
---

# Phase 2: MNIST digit recognition

Companion to `01-project-summary.md` (the contract, see its "Phase 2" section for goal + acceptance criteria) and `02-phase-00-and-01.md` (conventions to copy, do not invent new ones). This file turns that contract into a concrete work order.

Goal: first deep learning inference on the app. Task `p2-digit` takes a photo/drawing of a handwritten digit, preprocesses it to match MNIST (28x28, white strokes on black), runs a CNN trained by `app/cvlib/train.py`, and shows `top3` predictions plus the 28x28 image the model actually saw.

## Starting state

- The MNIST project code gets vendored into `backend/app/cvlib/` (`mnist.py` for inference, `train.py` as the standalone trainer); there is no `cvlib/models/` or `cvlib/data/` yet.
- Backend registers only `p1_opencv` (`main.py` line 11). `requirements.txt` has no torch/torchvision.
- Frontend renders arrays of objects in `data` as `key: value` rows (`DataValue` in `frontend/src/components/result-view.tsx` ~lines 20-37). No confidence-bar renderer exists yet.
- `p2-*` ids match no `ICON_RULES` regex in `task-sidebar.tsx` -> falls back to the generic `ScanSearch` icon (fine, or add a rule like `[/digit|classify/i, ...]`).

## Work items, in order

### 1. Project-side code: `backend/app/cvlib/`

Real logic lives in `cvlib/`; task files stay thin wrappers (convention 3). Mirror `imgtool.py`'s single-module style:

- `cvlib/mnist.py`: everything inference needs.
  - `DigitCNN` class (Conv 1->32, pool, Conv 32->64, pool; Linear 1600->128->10). The architecture must match the saved `state_dict` keys.
  - `preprocess(img_bgr, invert="auto") -> np.ndarray`: takes a BGR ndarray and returns the MNIST-shaped 28x28 (see pipeline below).
  - `load_model(weights_path) -> DigitCNN`: `torch.load(path, map_location="cpu", weights_only=True)`, `model.eval()`.
  - `predict(model, img28) -> list[dict]`: normalize, softmax, top-3 -> `[{"label": "7", "prob": 0.98}, ...]`.
- `cvlib/train.py`: standalone script (downloads MNIST into `cvlib/data/`, trains 3 epochs, prints test acc, saves `state_dict` to `cvlib/models/mnist_cnn.pt`). Keep it runnable on its own like the p1 CLI.

Modules import as `from app.cvlib import mnist`, a normal package import, so no `sys.path` juggling is needed.

### 2. Train the model (outside the app)

Training is a prerequisite, not part of the task handler (scope guard: no training inside the app). Run `train.py` once to produce `app/cvlib/models/mnist_cnn.pt` (~1.7 MB, ~99% test acc expected after 3 epochs on CPU). `cvlib/data/` will hold ~60 MB of MNIST raw files. Decide with the user who runs this step; the script itself is a deliverable either way.

### 3. Backend task: `backend/app/tasks/p2_mnist.py`

```python
WEIGHTS = Path(__file__).resolve().parents[1] / "cvlib" / "models" / "mnist_cnn.pt"
```

- `import` `app.cvlib.mnist` lazily (inside a cached `_get_model()`), not at module top: `import torch` costs ~1-3 s and weights must never load at import/registration time (convention 8). First `p2-digit` call pays the load; later calls reuse the module-level cache.
- Register task:

```python
@register_task(
    id="p2-digit",
    name="Recognize handwritten digit",
    project="mnist-digits",               # sidebar groups by this tag
    description="Classify a handwritten digit (0-9) with a CNN trained on MNIST.",
    params=[...invert select, see below...],
    output="image+data",
    result_tab="output",                  # the 28x28 view is the point of the task
    docs=DOCS_DIR / "p2-digit.md",
)
```

- Add `from app.tasks import p2_mnist  # noqa: F401` in `main.py`.

### 4. Params

Keep it minimal (convention 4). Recommended:

| Param    | Type   | Options / range          | Default | Why                                                |
| -------- | ------ | ------------------------ | ------- | -------------------------------------------------- |
| `invert` | select | `auto` / `always` / `never` | `auto` | Photos are usually dark-on-light; some inputs are already white-on-black. Same pattern as `p1-count`. |
| `pad`    | int    | 0-60, step 1             | 20      | Border around the cropped digit before resizing; MNIST digits never touch the edge. Group `"Preprocess"`. |

`auto` rule (same as `p1-count`): threshold with `THRESH_BINARY_INV + OTSU`, then `bitwise_not` when `white_ratio > 0.5` (means strokes were already white and INV flipped the wrong way).

### 5. Preprocessing pipeline (in `mnist.preprocess`)

The `preprocess()` implementation; input arrives as BGR ndarray:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
if invert == "always" or (invert == "auto" and (binary > 0).mean() > 0.5):
    binary = cv2.bitwise_not(binary)
coords = cv2.findNonZero(binary)
if coords is None:              # blank input: degrade, don't crash
    return cv2.resize(binary, (28, 28))
x, y, w, h = cv2.boundingRect(coords)
digit = binary[y:y + h, x:x + w]
side = max(w, h) + pad
canvas = np.zeros((side, side), np.uint8)
canvas[(side - h)//2:(side - h)//2 + h,
       (side - w)//2:(side - w)//2 + w] = digit
return cv2.resize(canvas, (28, 28))
```

### 6. Inference + outputs

```python
x = torch.tensor(img28, dtype=torch.float32).div(255)  # == ToTensor()
x = (x - 0.1307) / 0.3081                              # == Normalize((0.1307,), (0.3081,))
with torch.no_grad():
    probs = torch.softmax(model(x[None, None]), dim=1)[0]
top = probs.topk(3)
top3 = [{"label": str(int(i)), "prob": round(float(p), 4)}
        for p, i in zip(top.values, top.indices)]
```

- The normalize constants are the MNIST dataset mean/std; they must match training exactly or accuracy collapses.
- `image` output: `cv2.resize(img28, (280, 280), interpolation=cv2.INTER_NEAREST)` so the user sees the exact 28x28 tensor input. This is the main debugging view (per the docs: most inference bugs live in preprocessing).
- `data` output: `{"top3": top3}`. The `{label, prob}` object shape is the contract reused by `p3-sign` in phase 3.
- Inference on CPU only (`map_location="cpu"`): the model is tiny, CPU is deterministic, no MPS quirks.

### 7. Frontend: confidence bars in `result-view.tsx`

- Extend `DataValue`: when the value is an array whose objects all carry `label` and a numeric `prob` in [0, 1], render each row as label + horizontal bar (`width: prob * 100%`) + percentage text. Keep the existing `key: value` row rendering as the fallback for other shapes.
- Key on the data shape, never on `task.id` (convention 9); `p3-sign` will reuse this renderer unchanged.
- Optional: add a `digit.png` to `frontend/public/examples/` and append it to `EXAMPLES` in `drop-zone.tsx` (the list is shared across tasks, so keep the existing p1 samples first).

### 8. Dependencies

Add pinned `torch` and `torchvision` to `requirements.txt` (the file is a full `pip freeze` pin; install into `.venv` then re-freeze). The torch wheel is large (~hundreds of MB) so install once, before the demo.

### 9. Docs pair

`backend/app/tasks/docs/p2-digit.md` + `p2-digit.vi.md`, same structure as the p1 docs (convention 7): concept -> pipeline mermaid (photo -> gray -> otsu+invert -> crop -> pad -> 28x28 -> normalize -> CNN -> softmax top3) -> theory with KaTeX (softmax $p_i = e^{z_i}/\sum_j e^{z_j}$, why normalize with the MNIST constants, domain shift) -> the code -> param meanings -> tips (the "90% of bugs are preprocessing" idea, the blank-input case) -> "try it".

## Edge cases to handle

- **Blank/all-dark input**: `findNonZero` returns `None` -> the docs' version crashes. Return the degenerate 28x28 anyway (model will emit nonsense probs, which the top3 bars honestly show) or add a `data` note like `{"note": "no digit detected"}`. Pick one, document in the task docs.
- **Wrong polarity**: classic failure (white-on-black input treated as dark-on-light). `invert="always"/"never"` is the manual override; the displayed 28x28 image makes the bug visible.
- **First call slow**: torch import + weight load happen inside `_get_model()`, never at registration time.

## Verification loop

Same as phases 0-1 (convention 10):

1. `./dev.sh`; `curl localhost:8000/api/tasks` shows `p2-digit` with docs loaded.
2. `curl -F file=@digit.png -F params='{"invert":"auto"}' localhost:8000/api/tasks/p2-digit` -> check `data.top3` and decode the image.
3. First call slow (weight load), second call fast -> lazy loading works.
4. Playwright screenshot of the Output tab + top3 bars; test a drawn digit, a blank image, and `invert` overrides.
5. `npm run lint` (oxlint) + `tsc -b && vite build` clean.

## Acceptance criteria (from the contract)

- A self-drawn/photographed digit -> top-1 correct in most cases; the preprocessed image displays correctly (black background, white strokes).
- First request is slow (weight loading), subsequent requests are fast.

## As-built notes (filled in after implementation)

- Training ran on CPU: torch 2.15's MPS backend aborts on macOS 14 (`-[MPSNDArray arrayViewWithShape:strides:]`). `train.py` therefore defaults `--device cpu`; `mps`/`cuda`/`auto` remain available.
- Trained model: `app/cvlib/models/mnist_cnn.pt`, 0.9 MB, test acc 98.98% (hardest digits: 9 at 97.42%, 8 at 97.95%).
- Blank input returns the degenerate 28x28 plus `data.note = "no digit detected"` (option chosen: note).
- `digit.png` example generated from MNIST test image 0 (a 7); appended to `EXAMPLES` after the p1 samples.
- `p2-*` uses the generic sidebar icon (no `ICON_RULES` change).
- Verified end-to-end: `digit.png` -> 7 @ 99.8%; inverted (dark-on-light) copy -> still 7 @ 99.8% via `invert=auto`; blank -> note; first call ~2 s, later ~40 ms.
