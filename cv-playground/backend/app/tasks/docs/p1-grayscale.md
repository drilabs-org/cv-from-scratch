# Convert to grayscale (`p1-grayscale`)

Reduces a 3-channel color image to a single channel keeping only brightness. This is the opening step of almost every traditional image processing pipeline: `to_blur`, `to_edges`, `to_binary` and `count_objects` in `imgtool` all call `to_gray` internally first.

## Theory: what is brightness

Each pixel of a color image is a triple $(B, G, R)$. Grayscale compresses those 3 numbers into a single luminance using weights matching human perception:

$$Y = 0.299\,R + 0.587\,G + 0.114\,B \qquad \text{(ITU-R BT.601)}$$

Green weighs the most because the eye is most sensitive to it, blue the least. A plain average $(R+G+B)/3$ produces a "flat" image that loses the natural sense of brightness.

Three reasons grayscale leads every pipeline:

1. **3x less data**: `(H, W, 3)` -> `(H, W)`, so every later operation is faster and simpler.
2. **Many algorithms are only defined on one channel**: Canny, thresholding and contours all work on intensity.
3. **Removes color noise**: two pixels with the same brightness but different hue should not count as "different" when finding edges.

The trade-off: color information is gone for good. A bright red region and a green region with the same luminance map to the same gray level and can no longer be separated. Problems that need color (filtering objects by color, chroma key) must go through HSV + `inRange`, not grayscale.

## Tech stack: NumPy + OpenCV

- An image in OpenCV is a `numpy.ndarray` with `dtype=uint8` (0..255). A grayscale image has `shape = (H, W)`; a color image has `shape = (H, W, 3)`.
- Access pixels with `img[y, x]` (row first, column second), the opposite of the mathematical `(x, y)` intuition.
- The whole conversion is one function:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

- The constant is `COLOR_BGR2GRAY` because OpenCV reads images in **BGR** channel order (historical reason: Windows-era hardware when OpenCV was born). Using `COLOR_RGB2GRAY` by mistake swaps the red/blue weights and produces a gray image with an off tone.
- `cvtColor` uses fixed-point arithmetic, so multiplying the weights yourself in NumPy float gives results off by a few units; it does not affect the outcome.
- A 1-channel image saves to PNG fine, but **you cannot draw colors on it**: to annotate (the bbox in `p1-count`) you must convert back with `COLOR_GRAY2BGR`.

## Code

```python
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

Task `p1-grayscale` is just a wrapper calling this function, with no parameters.

## Tips & tricks

- Quick channel check: `img.ndim` (2 = gray, 3 = color) or `len(img.shape)`.
- A histogram via `cv2.calcHist([gray], [0], None, [256], [0, 256])` shows the brightness distribution: skewed left means underexposed, skewed right means overexposed, and two clear peaks signal that Otsu (`p1-binary`) will work well.
- Compute it yourself in NumPy to verify the weights: `(0.114*img[:,:,0] + 0.587*img[:,:,1] + 0.299*img[:,:,2]).astype(np.uint8)`.
- The `uint8` trap: addition and subtraction can wrap around (200 + 100 -> 44). For pixel arithmetic cast to `int16`/`float32`, or use `cv2.add` (saturates at 255 automatically).

## Try it

- Upload an image with a bright red region and a green region of equal brightness (traffic lights, flowers): both come out nearly the same gray. This is exactly why the later steps rely on intensity rather than color.
- Compared to the original, the gray image looks "older" because saturation is lost: that is the information that got compressed away.
