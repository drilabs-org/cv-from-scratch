# Spec: Recognize handwritten digit (`p2-digit`)

Demo reference: I/O contract, controls, safe ranges, and symptom fixes. Theory and pipeline explanation live in the English / Vietnamese tabs.

## I/O contract

- **Input**: any image `cv2.imdecode` can read, decoded as a BGR `uint8` array. Expects a single handwritten digit; dark-on-light (photo/ink) or white-on-black both work via `invert`.
- **Output**: `image+data`.
  - `image`: the preprocessed 28x28 tensor resized to 280x280 (`INTER_NEAREST`), i.e. exactly what the CNN saw. White strokes on black is correct polarity.
  - `data.top3`: `[{"label": "0"-"9", "prob": 0-1}, ...]`, rendered as confidence bars.
  - `data.note`: `"no digit detected"` when the thresholded frame is empty (blank input); `top3` is then meaningless by design.
- **Result tab**: `output` (set explicitly), because the preprocessed view is the debugging surface.
- **Model**: `backend/app/cvlib/models/mnist_cnn.pt` (`DigitCNN` state_dict, 0.9 MB), loaded lazily on first call and cached; CPU only. First request is slow (torch import + weight load), later ones are fast.

## Model selection

Set `CV_MODELS_DIR` to a directory containing `mnist_cnn.pt` to override the
bundled model. If the variable is unset or the file is missing, the bundled
model is used and the result view reports that choice. If a custom file exists
but is unreadable or incompatible, the task fails and the UI displays the
backend error; it does not silently fall back.

The custom artifact must be a `DigitCNN` `state_dict` with the same input
preprocessing, MNIST normalization, and 10 output classes. Model loading is
lazy and cached; restart the app after replacing the file.

## Parameters

| Param | Label | Type | Range | Default | Location | Pipeline stage |
| ----- | ----- | ---- | ----- | ------- | -------- | -------------- |
| `invert` | Invert polarity | select | `auto`, `always`, `never` | `auto` | main | polarity fix after Otsu INV |
| `pad` | Pad around digit (px) | int | 0 - 60, step 1 | 20 | Advanced: Preprocess | square canvas before resize |

## How to tune (adjust in this order)

1. **Look at the Output image first.** Correct input to the model = a white digit on black, centered with margin, not touching the edges. Everything else follows from that check.
2. **`invert`**: leave `auto` (flips when white covers > 50% of the thresholded frame). Use `always` when auto fails on a dark-on-light input, `never` for inputs already white-on-black.
3. **`pad`**: raise when the digit fills the whole 28x28 frame (MNIST digits never touch the edge); lower when the digit looks shrunken, e.g. a tall thin "1" padded into a wide square.

## Recommended presets

| Scenario | invert | pad |
| -------- | ------ | --- |
| `examples/digit.png`, `digit-3.png`, `digit-8.png` (MNIST test images) | auto | 20 |
| `examples/digit-photo.png` (simulated photo, dark on light) | auto | 20 |
| `examples/digit-hard.png` (ambiguous 1, reads as 7) | auto | 20 |
| Photo of ink/pencil on paper | auto | 20 |
| Whiteboard / tablet drawing (white on dark) | auto or never | 20 |
| Digit touches frame edges | auto | 30 - 40 |
| Very thin "1" looks distorted | auto | 10 - 15 |

## Symptoms -> fixes

- **Confidently wrong, Output shows black strokes on white**: polarity flipped the wrong way -> `invert=always` (or `never` if input was already white-on-black).
- **Digit stretched or squashed**: crop is non-square before padding is by design; if the digit still looks deformed, the bounding box clipped a stroke -> cleaner threshold needed (better lighting, thicker pen).
- **Low confidence across all 3 bars**: input is outside the training distribution (faint strokes, decorative font, motion blur). Thicker strokes, more contrast.
- **`no digit detected`**: blank or nearly-blank image; also happens when `invert` erases the strokes entirely.
- **First request takes seconds, then instant**: expected; weights load lazily and are cached, not a bug.

## Training reference

Weights come from `backend/app/cvlib/train.py` (run outside the app): MNIST train split auto-downloaded to `backend/app/cvlib/data/`, `CrossEntropyLoss` + `Adam(lr=0.001)`, batch 64, 3 epochs on CPU -> **test acc 98.98%** (9,898/10,000); hardest digits 9 (97.42%) and 8 (97.95%). Retrain: `cd backend/app/cvlib && python train.py --epochs 5`. Default `--device cpu` because torch 2.15 MPS crashed on macOS 14.
