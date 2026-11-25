# Count objects (`p1-count`)

The longest pipeline of phase 1, combining every concept from the 4 previous tasks: gray -> blur -> binary -> contours. It is also the only task that returns both an image and data (`data.count`). This page explains how the pipeline works; the **Spec** tab documents each demo control and its recommended values.

## Pipeline

```mermaid
flowchart LR
    A["BGR"] --> B["to_gray"]
    B --> C["to_blur<br/>ksize"]
    C --> D["threshold<br/>method + block + C"]
    D --> E["invert check<br/>white_ratio > 0.5"]
    E --> F["dilate<br/>dilate (optional)"]
    F --> G["findContours<br/>RETR_EXTERNAL"]
    G --> H["filter area > min_area"]
    H --> I["draw bbox + count"]
```

## Pipeline: stage by stage

### 1. `to_gray`: collapse color to luminance

- `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` folds 3 channels into one luminance channel: $Y = 0.299R + 0.587G + 0.114B$. Green weighs the most because the human eye is most sensitive to it.
- Threshold and contour algorithms only work on a **single channel**, and color is irrelevant for "object vs background" here, so dropping it early makes every later stage 3x cheaper.
- This stage is not called directly in `count_objects`: it runs inside `to_binary` via `to_blur`. No demo param controls it.

### 2. `to_blur`: Gaussian denoise

- `cv2.GaussianBlur(gray, (ksize, ksize), 0)` smooths the gray image before thresholding.
- Why it matters for counting specifically: every grain of noise that survives the threshold becomes a tiny white blob -> a fake contour -> `min_area` has to clean it up. Blur pushes those grains under the threshold so they never become blobs.
- Demo param: `ksize` (Pre-blur group). Odd, 1..31, default 5. Raising it also smears thin objects into the background and rounds off corners of real objects, so it trades detail for cleanliness.
- Like `to_gray`, this runs inside `to_binary`, so `ksize` is shared between this stage and the threshold stage.

### 3. Threshold: Otsu or adaptive

- `method=otsu`: `cv2.threshold(gray_blur, 0, 255, THRESH_BINARY + THRESH_OTSU)`. Otsu scans the histogram for the cut that maximizes between-class variance; the `thresh=0` argument is ignored. One global threshold for the whole image.
- `method=adaptive`: `cv2.adaptiveThreshold(gray_blur, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, block, C)`. Each pixel is compared against the Gaussian-weighted mean of its `block x block` neighborhood minus `C`, so the threshold follows local lighting.
- Demo params: `method` picks the branch; `block` (odd, 3..51) and `c` (-20..20) only apply when `method=adaptive`. Full math for both strategies is in the `p1-binary` doc.
- Output of this stage: a 1-channel `uint8` image containing only 0 and 255. **This is the quality gate**: every mistake left here becomes a counting error no later stage can fix.

### 4. `invert`: fix the polarity

- `findContours` counts **white** regions only, so objects must be white on a black background.
- `white_ratio = (binary > 0).mean()` is the fraction of white pixels. `invert=auto` flips the image with `cv2.bitwise_not` when white covers more than half the frame, on the heuristic that the larger region is usually the background. `always`/`never` override that guess manually.
- Failure mode of `auto`: when objects genuinely cover more than half the image (a dense pile of dark items on a light floor), the heuristic picks wrong and the background gets counted as one giant object.

### 5. `dilate`: bridge gaps (optional, skipped when `dilate=0`)

- `cv2.dilate(binary, kernel)` with a `MORPH_RECT` kernel of `dilate x dilate`: every white pixel grows to cover the whole kernel, so each white region expands roughly `dilate / 2` px per side.
- Two uses: (a) rejoin pieces of one object that the threshold split, e.g. a stroke broken in two; (b) intentionally merge neighboring objects to count groups instead of parts, like the `sample.png` "CV" trick.
- Cost: gaps smaller than ~`dilate` px between *different* objects close too, so it trades separation for completeness. One iteration only; repeat `dilate` calls would grow regions linearly.

### 6. `findContours`: boundary extraction

- `cv2.findContours(binary, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` returns one point array per white region: the contour is the ordered boundary pixels of that region.
- The binary image is consumed in place (OpenCV may modify it), which is safe here because nothing needs it afterward.
- Flag details and the measurement functions (`contourArea`, `boundingRect`, ...) are in the *Tech stack* section below.

### 7. `min_area`: final noise filter

- `kept = [c for c in contours if cv2.contourArea(c) > min_area]` drops speckle contours left over by noise and small blobs. It is the last denoising layer, applied to regions instead of pixels.
- `min_area` is the **main** demo control (always visible) because its right value scales with image resolution, not with the algorithm: it encodes "how big is one object in this particular image".

### 8. Annotate and return

- For each kept contour, `cv2.boundingRect` gives `(x, y, w, h)` and `cv2.rectangle` draws a green box on `img.copy()`.
- The function returns `(annotated_bgr, len(kept))`; the task wraps the count as `data.count`, and `result_tab="output"` makes the annotated view the default tab after a run.

## Theory: what is a contour

A contour is a curve hugging the boundary of a **connected white region** in a binary image. `findContours` scans the image and returns a list of contours, each an array of N points `(x, y)` ordered along the boundary.

Because the algorithm works on connected white regions:

- **Two objects touching each other** merge into one white region -> one contour -> under-count.
- **An object broken into pieces** in the binary image becomes several regions -> several contours -> over-count.

The key debugging principle: **counting quality is decided at the binarization step, not the contour step**. If contours look wrong, go back and fix the binary image (more blur, different threshold, morphology); do not fix the contours.

## Tech stack: OpenCV

```python
contours, hierarchy = cv2.findContours(binary,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
```

- `RETR_EXTERNAL`: only outermost boundaries, ignoring inner holes and nested regions (a coin with a hole still counts as 1). Use `RETR_TREE` or `RETR_LIST` if you need the parent-child hierarchy.
- `CHAIN_APPROX_SIMPLE`: compresses straight runs into just their endpoints instead of storing every pixel, making contours much lighter; enough for `contourArea` and `boundingRect`. Use `CHAIN_APPROX_NONE` if you need every boundary point.
- OpenCV 4 returns `(contours, hierarchy)`; old OpenCV 3 snippets online return 3 values and will fail to unpack.

Measurements per contour:

| Function | Returns | Used for |
|-----|--------|---------|
| `cv2.contourArea(c)` | pixel area inside | filtering noise, measuring size |
| `cv2.boundingRect(c)` | `(x, y, w, h)` upright box | drawing bbox, cropping regions |
| `cv2.arcLength(c, True)` | perimeter | shape features |
| `cv2.approxPolyDP(c, eps, True)` | approximating polygon | shape recognition (3 vertices = triangle...) |
| `cv2.minEnclosingCircle(c)` | center + radius | round objects, tracking |
| `cv2.moments(c)` | statistical moments | centroid $c_x = m_{10} / m_{00}$ |

### Drawing the detection

- `cv2.boundingRect(c)` -> upright box `(x, y, w, h)`, drawn with `cv2.rectangle` (what this task uses). Fast and simple, but always axis-aligned: a rotated object gets a loose, oversized box.
- `cv2.minAreaRect(c)` -> rotated box `((cx, cy), (w, h), angle)`, the smallest rectangle at any orientation. No direct draw helper: convert with `box = cv2.boxPoints(rect)`, round via `np.int0`, then `cv2.drawContours(out, [box], 0, color, 2)`.
- `cv2.minEnclosingCircle(c)` -> `(center, radius)` for `cv2.circle`: better for round or blobby objects where a rectangle looks wrong.
- `cv2.drawContours(out, kept, -1, color, 2)` skips boxes entirely and draws the true contour polygon: best when the exact shape matters.

## Code

```python
def count_objects(img, min_area=500, ksize=5, method="otsu", invert="auto",
                  dilate=0, block=11, C=2):
    binary = to_binary(img, method, ksize, block, C)
    white_ratio = (binary > 0).mean()
    if invert == "always" or (invert == "auto" and white_ratio > 0.5):
        binary = cv2.bitwise_not(binary)
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
        binary = cv2.dilate(binary, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    kept = [c for c in contours if cv2.contourArea(c) > min_area]
    for c in kept:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out, len(kept)
```

- The task's params map onto the pipeline stages: `ksize` -> `to_blur`, `method` -> Otsu/adaptive branch of `to_binary`, `block`/`c` -> adaptive blockSize/C (only used when `method=adaptive`), `invert` -> binary polarity, `dilate` -> optional morphology bridging nearby regions into one contour, `min_area` -> final contour filter. Only `min_area` stays always visible; the rest sit under Advanced grouped by stage (Pre-blur / Threshold / Morphology).
- `dilate` grows every white region by about `dilate / 2` px per side, so regions separated by less than ~`dilate` px merge. On `sample.png` the letters "CV" are ~9px apart: `dilate=9` counts them as one object while the square and circle stay separate.
- `findContours` only counts **white** regions, so objects must be white on a black background. `invert="auto"` flips when white covers more than half the image (white is the background); `always`/`never` force it manually. Without this, a light-background image like `sample.png` counts 1 "object": the whole background.
- `out = img.copy()` is critical: `cv2.rectangle` **draws in-place**, and `ndarray` is passed by reference, so without copying you would mutate the caller's original image.
- Draw on `img.copy()` (the BGR color image), not the binary image: a 1-channel image cannot hold the color `(0, 255, 0)`.
- `count = len(kept)` after filtering, not the total contour count: `min_area` is the pipeline's final denoising layer.

## Tips & tricks

- `min_area` depends on image size: 500px is reasonable for ~1MP images but too large for thumbnails. Rule of thumb: set it around 0.01% - 0.1% of the image area.
- **Two touching objects = one contour**: the biggest limitation of this method. Mitigate with `morphologyEx(MORPH_OPEN)` to split thin joins, or distance transform + watershed for heavily stuck objects (stacked coins).
- Objects broken apart by a gappy binary image: `MORPH_CLOSE` or a light dilate before `findContours` to bridge the gaps.
- Quick debug: `cv2.drawContours(out, kept, -1, (0,0,255), 2)` draws the actual contour shape instead of boxes, making bad contours easy to spot.
- The color `(0, 255, 0)` is in BGR order so it comes out green; the wrong order draws the wrong color.

## Try it

- `coins.jpg` + `min_area=500`: accurate count. Raise it to 5000: small coins disappear. Drop it to 0: background noise gets counted as objects.
- An image with touching objects (two coins edge to edge): witness the under-count. This is the motivation for learning morphology and watershed in later projects.
