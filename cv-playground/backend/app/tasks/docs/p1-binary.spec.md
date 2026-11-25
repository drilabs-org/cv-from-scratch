# Spec: Binarization (`p1-binary`)

Demo reference: what each control does, its safe range, and recommended starting points. Theory and pipeline explanation live in the English / Vietnamese tabs.

## I/O contract

- **Input**: any image `cv2.imdecode` can read, decoded as BGR `uint8`. Internally `to_gray` -> `GaussianBlur(ksize)` before thresholding.
- **Output**: 1-channel binary PNG (`output="image"`) containing only values 0 and 255. No `data`.
- **Result tab**: `auto` -> lands on Output.

## Parameters

| Param | Label | Type | Range | Default | Location |
| ----- | ----- | ---- | ----- | ------- | -------- |
| `method` | Method | select | `otsu`, `adaptive` | `otsu` | main |
| `ksize` | Blur kernel (odd) | int | 1 - 31, step 2 | 5 | Advanced: Pre-blur |
| `block` | Block size (odd) | int | 3 - 51, step 2 | 11 | Advanced: Adaptive |
| `c` | Constant C | float | -20 - 20, step 0.5 | 2 | Advanced: Adaptive |

`block` and `c` only apply when `method=adaptive`; they do nothing under `otsu`.

## How to tune

- **`method=otsu`**: pick when lighting is uniform and the histogram is bimodal (clear object vs background peaks). Zero knobs to tune; always start here.
- **`method=adaptive`**: pick when lighting varies across the frame: shadows, a page photographed with one side lit, vignetting.
- **`block`**: neighborhood size for the local threshold. It must be odd and should exceed the feature size you want to keep: use ~1.5 - 3x the stroke/blob width in pixels. Too small makes the threshold chase noise; too large degenerates back toward a global threshold.
- **`c`**: subtracted from the local mean. Positive values make the threshold stricter (fewer white pixels, thinner strokes); negative values make it more permissive. Typical document range: 2 - 10.
- **`ksize`**: denoise before cutting the threshold. Raise it when the binary is speckled, but note blur can also break weak strokes under `adaptive`.

## Recommended presets

| Scenario | method | block | c | ksize |
| -------- | ------ | ----- | - | ----- |
| `coins.jpg` / studio shot | otsu | - | - | 5 |
| Scanned page with lighting gradient | adaptive | 15 - 25 | 5 - 10 | 5 |
| Thin handwriting strokes | adaptive | 9 - 13 | 2 - 4 | 3 |
| Speckled, noisy binary | otsu | - | - | 9 |

## Symptoms -> fixes

- **One half of the image all black (shadow side lost)**: switch to `adaptive` with `block` ~21.
- **Strokes too fat or broken**: adjust `c` first (fat -> raise `c`, broken -> lower `c`) before touching `block`.
- **Object black on white background**: the binary polarity is inverted. This demo has no polarity control; `p1-count` handles it via `invert`.
