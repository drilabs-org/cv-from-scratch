# Gaussian blur (`p1-blur`)

Convolution with a Gaussian kernel: each pixel is replaced by a weighted average of its neighborhood, with the center weighing the most. In real pipelines this is the **intermediate denoising step**: `to_edges`, `to_binary` and `count_objects` all call `to_blur` before processing.

## Place in the pipeline

```mermaid
flowchart LR
    A["BGR"] --> B["to_gray"]
    B --> C["GaussianBlur k x k"]
    C --> D["Canny / threshold<br/>(next step)"]
```

## Theory: convolution and the Gaussian kernel

Image filtering = sliding a small matrix (kernel) over every image position; each new pixel is a weighted sum of the pixels covered by the kernel. The Gaussian kernel is generated from a 2D bell function:

$$G(x, y) = \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

Pixels near the center get high weights, weights fall off toward 0 with distance. Unlike a uniform box kernel (`cv2.blur`, every position is $1/k^2$), Gaussian "trusts" nearby pixels more, so it smooths naturally and creates less blockiness.

In frequency terms, Gaussian is a **low-pass filter**: it suppresses high-frequency intensity oscillations (grain noise, fine strokes) while keeping low-frequency structure (uniform regions, large blobs). That is why thresholding and Canny must come after blur: otherwise every noise grain becomes an "object" or a fake edge segment.

The trade-off: blur also erases real detail at the same high frequencies, and smears edges so Canny localizes them less precisely when blur is too strong. Choosing ksize is choosing the balance point between clean noise removal and preserved structure.

## Tech stack: OpenCV

```python
blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
```

- `ksize`: side of the square kernel, **must be odd** so there is a single center pixel; `GaussianBlur` errors on even values. The task limits it to 1..31.
- `sigmaX = 0`: tells OpenCV to derive $\sigma$ from ksize via $\sigma \approx 0.3 \cdot ((k-1) \cdot 0.5 - 1) + 0.8$. With `ksize=5`, $\sigma \approx 1.1$. The task exposes this as the advanced param `sigma`: raise it for a wider bell without growing the kernel.
- The Gaussian kernel is **separable**: it splits into two 1D passes (horizontal then vertical), so cost is $O(H \cdot W \cdot k)$ instead of $O(H \cdot W \cdot k^2)$. That is why large ksize still runs fine on CPU.
- Border pixels lack neighbors: OpenCV defaults to `BORDER_DEFAULT` (reflect `gfedcb|abcdefgh|gfedcba`), so edges do not go dark the way they do with zero padding.

Comparing blur filters:

| Function | Mechanism | Best at |
|-----|--------|--------|
| `cv2.blur` | box average | fastest, cheapest blur |
| `cv2.GaussianBlur` | Gaussian weights | standard denoise before Canny/threshold |
| `cv2.medianBlur` | neighborhood median | salt-and-pepper noise, keeps edges better |
| `cv2.bilateralFilter` | two-sided filter (space + intensity) | smooths while preserving edges, much slower |

## Code

```python
def to_blur(img, ksize=5, sigma=0):
    return cv2.GaussianBlur(to_gray(img), (ksize, ksize), sigma)
```

- Blur runs **on the grayscale image**, so the task output is a 1-channel image even for color input.
- The task's `ksize` parameter (1..31, default 5) must be odd; the handler forces `+1` if it receives an even value.
- `ksize` is the main control; `sigma` sits under Advanced (0 = auto-derive from ksize).

## Tips & tricks

- Pick ksize by purpose: 3-5 for light denoising before threshold/Canny; 9-15 to blur out a background; above 21 is mostly a visual effect.
- Salt-and-pepper noise (scattered white/black dots): Gaussian smears the noise instead of removing it; `medianBlur` is the right tool.
- Want smooth skin/background while keeping sharp edges (retouching): use `bilateralFilter`, it only blurs pixels with similar intensity.
- Convolution is the foundation of CNNs too: here the kernel is chosen by you, while kernels in a CNN Conv layer are learned by the model. Understanding `filter2D` is half of understanding a Conv layer.

## Try it

- Drag `ksize` from 1 -> 31 on the same image: object edges progressively smear. Then run `p1-edges` to see Canny misplace and break edges when blur is too heavy.
- On a heavily noisy image: watch Gaussian fail to fully clean it, which is exactly why `medianBlur` exists.
