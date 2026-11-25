# Spec: Count objects (`p1-count`)

Demo reference: what each control does, its safe range, and recommended starting points. Theory and pipeline explanation live in the English / Vietnamese tabs.

## I/O contract

- **Input**: any image `cv2.imdecode` can read (JPG, PNG, BMP, WebP...), decoded as a BGR `uint8` array.
- **Output**: `image+data`. The image is the original BGR frame with a green `cv2.rectangle` per detected object; `data.count` is the number of kept contours, rendered as a badge.
- **Result tab**: `output` (set explicitly via `result_tab="output"`), because the annotated image is the point of the demo. Input and Compare remain available.

## Parameters

| Param | Label | Type | Range | Default | Location | Pipeline stage |
| ----- | ----- | ---- | ----- | ------- | -------- | -------------- |
| `min_area` | Minimum area (px) | int | 0 - 20000 | 500 | main | contour filter |
| `ksize` | Blur kernel (odd) | int | 1 - 31, step 2 | 5 | Advanced: Pre-blur | `to_blur` inside `to_binary` |
| `method` | Threshold method | select | `otsu`, `adaptive` | `otsu` | Advanced: Threshold | threshold branch |
| `invert` | Invert binary | select | `auto`, `always`, `never` | `auto` | Advanced: Threshold | polarity fix |
| `block` | Adaptive block (odd) | int | 3 - 51, step 2 | 11 | Advanced: Threshold | adaptive only |
| `c` | Adaptive C | float | -20 - 20, step 0.5 | 2 | Advanced: Threshold | adaptive only |
| `dilate` | Merge gap (dilate px) | int | 0 - 31 | 0 | Advanced: Morphology | optional morphology |

`block` and `c` are ignored unless `method=adaptive`. `dilate=0` skips morphology entirely.

## How to tune (adjust in this order)

1. **Fix the binary first.** Counting quality is decided at the threshold stage; `findContours` only reports what the binary gives it. Check symptoms below before touching `min_area`.
2. **`min_area`**: expected pixel area of one object. Rule of thumb: ~0.01 - 0.1% of the image area. 500 fits ~1MP images; scale down for thumbnails, up for 4K.
3. **`invert`**: leave `auto` (flips when white covers > 50% of the frame). Use `always` for dark objects on a light background when auto misjudges, `never` for light-on-dark.
4. **`dilate`**: only to rejoin a fragmented object, or to merge intentional groups (the "CV" trick). Set it to roughly the gap in pixels you want to bridge. Every region grows ~`dilate/2` px per side, so different objects closer than ~`dilate` px merge too.
5. **`ksize`**: raise when noise specks survive into contours; lower when blur eats thin objects.
6. **`method` / `block` / `c`**: same tuning as `p1-binary`. Use `adaptive` only for uneven lighting; `block` ~1.5 - 3x the stroke width; positive `c` thins white regions.

## Recommended presets

| Scenario | method | invert | min_area | dilate | ksize |
| -------- | ------ | ------ | -------- | ------ | ----- |
| `coins.jpg` (clean, uniform light) | otsu | auto | 500 | 0 | 5 |
| `sample.png` (white background) | otsu | auto | 200 - 500 | 0 | 5 |
| "CV" letters counted as one object | otsu | auto | 500 | 9 | 5 |
| Uneven lighting / shadows | adaptive | auto | scaled to image | 0 - 5 | 5 |
| Objects fragmented by threshold | otsu | auto | scaled to image | 3 - 9 | 5 - 9 |
| Thumbnail / small image | otsu | auto | 20 - 100 | 0 | 3 - 5 |

## Symptoms -> fixes

- **Count = 1 on a light-background photo**: polarity is wrong; try `invert=always`. Auto should already catch this, so if it did not, objects cover more than half the frame.
- **Count too high, tiny boxes everywhere**: raise `min_area`, or raise `ksize` so specks never reach the threshold.
- **Two touching objects counted as 1**: inherent limit of connected-component counting; this demo cannot split them (needs `MORPH_OPEN` or watershed). Make sure `dilate=0`, it makes merging worse.
- **One object counted as several**: `dilate` 3 - 9 to bridge the gaps, or stronger `ksize` before thresholding.
- **Objects disappear entirely**: `min_area` too high, or `ksize` so large that thin objects blurred into the background.
