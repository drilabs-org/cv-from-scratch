# Spec: Gaussian blur (`p1-blur`)

Demo reference: what each control does, its safe range, and recommended starting points. Theory and pipeline explanation live in the English / Vietnamese tabs.

## I/O contract

- **Input**: any image `cv2.imdecode` can read, decoded as BGR `uint8`.
- **Output**: blurred **grayscale** PNG (`output="image"`), no `data`. Note the output is 1-channel even for color input: `to_blur` runs on the `to_gray` result.
- **Result tab**: `auto` -> lands on Output. The Compare slider is a good way to show progressive smearing.

## Parameters

| Param | Label | Type | Range | Default | Location |
| ----- | ----- | ---- | ----- | ------- | -------- |
| `ksize` | Kernel size (odd) | int | 1 - 31, step 2 | 5 | main |
| `sigma` | Sigma (0 = auto) | float | 0 - 10, step 0.1 | 0 | Advanced: Gaussian |

## How to tune

- **`ksize`** is the blur radius. It must be odd so the kernel has a center pixel; the handler bumps even values +1, so typing 6 behaves as 7.
  - `1`: effectively disables the filter (kernel = single pixel). Handy to show the "before" state without switching tasks.
  - `3 - 5`: light denoising, what downstream tasks (`p1-edges`, `p1-binary`, `p1-count`) assume.
  - `9 - 15`: clearly visible smoothing; flattens skin/background texture.
  - `21+`: mostly an artistic blur; real detail is gone.
- **`sigma`** controls the bell width directly. `0` (default) derives it from ksize: $\sigma \approx 0.3 \cdot ((k-1) \cdot 0.5 - 1) + 0.8$, about 1.1 at `k=5`. Set it manually when you want a wider bell without growing the kernel, or a tighter bell inside a large kernel.

## Recommended presets

| Scenario | ksize | sigma |
| -------- | ----- | ----- |
| Pre-threshold denoise (default) | 5 | 0 |
| Strong sensor noise | 9 - 13 | 0 |
| Blur out a busy background | 15 - 25 | 0 |
| Wide bell, small kernel (fine control) | 5 | 2 - 4 |

## Symptoms -> fixes

- **Output looks unchanged**: you are on `ksize=1`.
- **Edges smear so much that `p1-edges` breaks them**: reduce `ksize` to 3 - 5.
- **Salt-and-pepper specks survive**: Gaussian is the wrong filter for that noise type; `medianBlur` fits better (not exposed in this demo).
