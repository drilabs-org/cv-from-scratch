"""Pixel-value distribution of the 60k MNIST train images, before and after
the Normalize((0.1307,), (0.3081,)) transform from train.py.

Output: one PNG with two side-by-side histograms (log y-scale, because
~80% of MNIST pixels are black background and would drown the rest).

    cd practices/02-mnist-chu-so-viet-tay/src/deep-dive
    python plot_normalize_distribution.py
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from torchvision import datasets  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent
                    / "normalize_distribution.png")
    args = ap.parse_args()

    train_set = datasets.MNIST(str(DATA_DIR), train=True, download=True)
    pixels = train_set.data.numpy().ravel()  # uint8, 60000*784 ~ 47M values
    n_img, n_px = train_set.data.shape[0], pixels.size
    print(f"train: {n_img} imgs | {n_px:,} pixels | "
          f"zero px: {(pixels == 0).mean():.1%}")

    normalized = (pixels.astype(np.float32) / 255.0 - MNIST_MEAN) / MNIST_STD

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_raw.hist(pixels, bins=64, range=(0, 255), color="steelblue")
    ax_raw.set_title(f"Before normalize: raw pixels [0, 255]\n"
                     f"mean {pixels.mean():.1f} | std {pixels.std():.1f}")
    ax_raw.set_xlabel("Pixel value (uint8)")
    ax_raw.set_ylabel("Pixel count (log scale)")
    ax_raw.set_yscale("log")

    ax_norm.hist(normalized, bins=64, color="darkorange")
    ax_norm.set_title(f"After normalize: (x/255 - {MNIST_MEAN}) / {MNIST_STD}\n"
                      f"mean {normalized.mean():.3f} | "
                      f"std {normalized.std():.3f} | "
                      f"range [{normalized.min():.2f}, {normalized.max():.2f}]")
    ax_norm.set_xlabel("Normalized value")
    ax_norm.set_yscale("log")

    fig.suptitle(f"Pixel value distribution, MNIST train set ({n_img:,} images)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out} ({args.out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
