"""Tasks for project 1 (opencv-basics): thin wrappers over cvlib.imgtool."""
from pathlib import Path

from app.cvlib import imgtool
from app.registry import register_task

PROJECT = "opencv-basics"
DOCS_DIR = Path(__file__).resolve().parent / "docs"


@register_task(
    id="p1-grayscale",
    name="Convert to grayscale",
    project=PROJECT,
    description="Convert a BGR color image to a single-channel grayscale image.",
    output="image",
    docs=DOCS_DIR / "p1-grayscale.md",
)
def p1_grayscale(img, params):
    return imgtool.to_gray(img), {}


@register_task(
    id="p1-blur",
    name="Gaussian blur",
    project=PROJECT,
    description="Blur the image with a Gaussian filter to reduce noise before further processing.",
    params=[
        {
            "name": "ksize",
            "type": "int",
            "min": 1,
            "max": 31,
            "step": 2,
            "default": 5,
            "label": "Kernel size (odd)",
        },
        {
            "name": "sigma",
            "type": "float",
            "min": 0,
            "max": 10,
            "step": 0.1,
            "default": 0,
            "label": "Sigma (0 = auto)",
            "group": "Gaussian",
        },
    ],
    output="image",
    docs=DOCS_DIR / "p1-blur.md",
)
def p1_blur(img, params):
    ksize = params.get("ksize", 5)
    if ksize % 2 == 0:
        ksize += 1
    return imgtool.to_blur(img, ksize, params.get("sigma", 0)), {}


@register_task(
    id="p1-edges",
    name="Canny edge detection",
    project=PROJECT,
    description="Find edges in the image using the Canny algorithm.",
    params=[
        {
            "name": "t1",
            "type": "int",
            "min": 0,
            "max": 255,
            "default": 50,
            "label": "Low threshold",
        },
        {
            "name": "t2",
            "type": "int",
            "min": 0,
            "max": 255,
            "default": 150,
            "label": "High threshold",
        },
        {
            "name": "ksize",
            "type": "int",
            "min": 1,
            "max": 31,
            "step": 2,
            "default": 5,
            "label": "Blur kernel (odd)",
            "group": "Pre-blur",
        },
        {
            "name": "aperture",
            "type": "int",
            "min": 3,
            "max": 7,
            "step": 2,
            "default": 3,
            "label": "Sobel aperture",
            "group": "Canny",
        },
        {
            "name": "l2",
            "type": "select",
            "options": ["l1", "l2"],
            "default": "l1",
            "label": "Gradient norm",
            "group": "Canny",
        },
    ],
    output="image",
    docs=DOCS_DIR / "p1-edges.md",
)
def p1_edges(img, params):
    ksize = params.get("ksize", 5)
    if ksize % 2 == 0:
        ksize += 1
    return (
        imgtool.to_edges(
            img,
            params.get("t1", 50),
            params.get("t2", 150),
            ksize=ksize,
            aperture=params.get("aperture", 3),
            l2=params.get("l2", "l1") == "l2",
        ),
        {},
    )


@register_task(
    id="p1-binary",
    name="Binarization",
    project=PROJECT,
    description="Convert a grayscale image to binary (black/white only).",
    params=[
        {
            "name": "method",
            "type": "select",
            "options": ["otsu", "adaptive"],
            "default": "otsu",
            "label": "Method",
        },
        {
            "name": "ksize",
            "type": "int",
            "min": 1,
            "max": 31,
            "step": 2,
            "default": 5,
            "label": "Blur kernel (odd)",
            "group": "Pre-blur",
        },
        {
            "name": "block",
            "type": "int",
            "min": 3,
            "max": 51,
            "step": 2,
            "default": 11,
            "label": "Block size (odd)",
            "group": "Adaptive",
        },
        {
            "name": "c",
            "type": "float",
            "min": -20,
            "max": 20,
            "step": 0.5,
            "default": 2,
            "label": "Constant C",
            "group": "Adaptive",
        },
    ],
    output="image",
    docs=DOCS_DIR / "p1-binary.md",
)
def p1_binary(img, params):
    ksize = params.get("ksize", 5)
    block = params.get("block", 11)
    if ksize % 2 == 0:
        ksize += 1
    if block % 2 == 0:
        block += 1
    return (
        imgtool.to_binary(
            img,
            params.get("method", "otsu"),
            ksize=ksize,
            block=block,
            C=params.get("c", 2),
        ),
        {},
    )


@register_task(
    id="p1-count",
    name="Count objects",
    project=PROJECT,
    description="Count objects via contours on the binary image, drawing bounding boxes.",
    params=[
        {
            "name": "min_area",
            "type": "int",
            "min": 0,
            "max": 20000,
            "default": 500,
            "label": "Minimum area (px)",
        },
        {
            "name": "ksize",
            "type": "int",
            "min": 1,
            "max": 31,
            "step": 2,
            "default": 5,
            "label": "Blur kernel (odd)",
            "group": "Pre-blur",
        },
        {
            "name": "method",
            "type": "select",
            "options": ["otsu", "adaptive"],
            "default": "otsu",
            "label": "Threshold method",
            "group": "Threshold",
        },
        {
            "name": "invert",
            "type": "select",
            "options": ["auto", "always", "never"],
            "default": "auto",
            "label": "Invert binary",
            "group": "Threshold",
        },
        {
            "name": "block",
            "type": "int",
            "min": 3,
            "max": 51,
            "step": 2,
            "default": 11,
            "label": "Adaptive block (odd)",
            "group": "Threshold",
        },
        {
            "name": "c",
            "type": "float",
            "min": -20,
            "max": 20,
            "step": 0.5,
            "default": 2,
            "label": "Adaptive C",
            "group": "Threshold",
        },
        {
            "name": "dilate",
            "type": "int",
            "min": 0,
            "max": 31,
            "default": 0,
            "label": "Merge gap (dilate px)",
            "group": "Morphology",
        },
    ],
    output="image+data",
    result_tab="output",
    docs=DOCS_DIR / "p1-count.md",
)
def p1_count(img, params):
    ksize = params.get("ksize", 5)
    block = params.get("block", 11)
    if ksize % 2 == 0:
        ksize += 1
    if block % 2 == 0:
        block += 1
    out, n = imgtool.count_objects(
        img,
        min_area=params.get("min_area", 500),
        ksize=ksize,
        method=params.get("method", "otsu"),
        invert=params.get("invert", "auto"),
        dilate=params.get("dilate", 0),
        block=block,
        C=params.get("c", 2),
    )
    return out, {"count": n}
