# Spec: Canny edge detection (`p1-edges`)

Demo reference: what each control does, its safe range, and recommended starting points. Theory and pipeline explanation live in the English / Vietnamese tabs.

## I/O contract

- **Input**: any image `cv2.imdecode` can read, decoded as BGR `uint8`. Internally converted to grayscale and blurred first.
- **Output**: 1-channel edge map PNG (`output="image"`): white pixels on edges, black elsewhere. No `data`.
- **Result tab**: `auto` -> lands on Output. Compare is useful to see which image structures became edges.

## Parameters

| Param | Label | Type | Range | Default | Location |
| ----- | ----- | ---- | ----- | ------- | -------- |
| `t1` | Low threshold | int | 0 - 255 | 50 | main |
| `t2` | High threshold | int | 0 - 255 | 150 | main |
| `ksize` | Blur kernel (odd) | int | 1 - 31, step 2 | 5 | Advanced: Pre-blur |
| `aperture` | Sobel aperture | int | 3 - 7, step 2 | 3 | Advanced: Canny |
| `l2` | Gradient norm | select | `l1`, `l2` | `l1` | Advanced: Canny |

## How to tune

- Canny uses **hysteresis**: gradients above `t2` are edges, below `t1` are not, and in-between gradients survive only if connected to a strong edge. So `t1` controls continuity (fewer broken segments) and `t2` controls selectivity (fewer weak edges).
- Keep `t2` roughly 2 - 3x `t1` (the 50/150 default). Setting `t1 >= t2` degenerates hysteresis into a plain hard threshold.
- **Low-contrast scene (faint edges)**: lower both, e.g. 20/60. **Busy or high-contrast scene**: raise both, e.g. 80/200.
- **`ksize`** (pre-blur): raise to 7 - 9 on noisy images or the edge map fills with speckles; lower to 3 for sharper edge localization on clean images.
- **`aperture`** is the Sobel kernel size used for gradients: 5 or 7 smooths the gradient estimate (fewer fragmented edges, worse localization); 3 is the standard choice.
- **`l2`**: `l2` uses the true magnitude $\sqrt{g_x^2 + g_y^2}$, `l1` uses $|g_x| + |g_y|$ which is faster but less accurate. Mostly a teaching toggle; the visual difference is subtle.

## Recommended presets

| Scenario | t1 | t2 | ksize |
| -------- | -- | -- | ----- |
| Clean photo, balanced detail | 50 | 150 | 5 |
| High contrast / heavy texture | 80 | 200 | 5 |
| Low contrast, faint outlines | 20 | 60 | 3 |
| Noisy image | 70 | 180 | 9 |

## Symptoms -> fixes

- **Edge map full of speckles**: raise `ksize` first, then `t1`.
- **Edges of real objects missing**: lower `t2`; if they return fragmented, lower `t1` too.
- **Edges look doubled or displaced**: `aperture` too large or the input is already blurry; drop to `aperture=3` and `ksize=3`.
