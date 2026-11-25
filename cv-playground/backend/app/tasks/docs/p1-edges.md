# Canny edge detection (`p1-edges`)

Returns a binary image: white = edge, black = not edge. This is the first "analysis" task: instead of transforming the image, it extracts **structure** (object boundaries) out of it.

## Place in the pipeline

```mermaid
flowchart LR
    A["BGR"] --> B["to_gray"]
    B --> C["to_blur k=5"]
    C --> D["cv2.Canny t1 t2"]
```

## Theory: Canny in 4 steps

1. **Gaussian blur** to remove noise (in the code: `to_blur` with ksize=5).
2. **Sobel** computes the gradient: partial derivatives along $x$ ($G_x$) and $y$ ($G_y$) via two convolution kernels, e.g.

   $$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$

   From these come magnitude $|G|$ and direction $\theta = \operatorname{atan2}(G_y, G_x)$ at each pixel. Edges live where intensity changes abruptly, i.e. where $|G|$ is large.
3. **Non-maximum suppression**: along the gradient direction, keep only the pixel with the largest $|G|$ in its neighborhood. Edges shrink from thick bands down to 1px.
4. **Hysteresis threshold** with 2 thresholds:
   - $|G| > t_2$: strong edge, definitely kept.
   - $|G| < t_1$: discarded.
   - Between the two: weak edge, kept only if **connected** to a strong edge.

Hysteresis is Canny's smartest idea: a real edge that is weak in a few stretches still gets linked through, while isolated noise is discarded. A single threshold cannot tell these two cases apart.

Because derivatives amplify high-frequency noise, the blur in step 1 is **not optional**: skip it and every noise grain produces a strong gradient response, flooding the result with fake edges.

## Tech stack: OpenCV

```python
edges = cv2.Canny(gray_blur, threshold1=t1, threshold2=t2)
```

- The actual input has already been through `to_gray` + `to_blur` inside `imgtool`; `cv2.Canny` takes a 1-channel image.
- The output is a 1-channel image containing only 0 and 255.
- Advanced params worth trying: `aperture` maps to `apertureSize` (Sobel kernel size, 3/5/7, default 3) and `l2` maps to `L2gradient` (`l1` uses $|G_x| + |G_y|$, fast; `l2` uses $\sqrt{G_x^2 + G_y^2}$, more accurate, slightly slower).
- Canny returns **edges, not object regions**: edges are usually open contours so do not call `findContours` directly to count objects. Go through thresholding like `p1-count` does.

## Code

```python
def to_edges(img, t1=50, t2=150, ksize=5, aperture=3, l2=False):
    return cv2.Canny(to_blur(img, ksize), t1, t2,
                     apertureSize=aperture, L2gradient=l2)
```

The main controls are `t1`, `t2` (0..255, default 50/150). Everything else sits under Advanced, grouped by pipeline stage: `ksize` tunes the pre-blur (Pre-blur group), `aperture` and `l2` tune the Sobel gradient step inside Canny (Canny group).

## Tips & tricks

- Rule of thumb $t_2 : t_1 \approx 2:1$ to $3:1$ (default 50/150). If edges are missing, lower `t1` first because `t1` controls how much of the weak stretches gets linked; for noisy images raise both.
- **Auto-Canny** with no manual thresholds: take $v = \mathrm{median}(\text{gray image})$, set $t_1 = 0.66\,v$, $t_2 = 1.33\,v$, clip to 0..255. Useful for batch-processing images with varying brightness.
- `t1 >= t2` still runs but yields almost no edges: OpenCV does not enforce ordering between the two thresholds.
- Canny is the first step of `HoughLinesP`/`HoughCircles` (line and circle detection) and of industrial dimension-measurement pipelines.

## Try it

- Keep `t2=150`, drag `t1` from 0 up to 140: the number of linked edge segments drops sharply as `t1` rises, showing the role of the "weak edge" band.
- Keep `t1=50`, drop `t2` to 60: noisy edges appear everywhere because the "strong" threshold is too easy to reach. This is why blur must come before Canny.
