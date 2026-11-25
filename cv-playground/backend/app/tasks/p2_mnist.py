"""Tasks for project 2 (mnist-digits): MNIST digit CNN."""
from pathlib import Path

import cv2

from app.registry import register_task

PROJECT = "mnist-digits"
DOCS_DIR = Path(__file__).resolve().parent / "docs"
WEIGHTS = Path(__file__).resolve().parents[1] / "cvlib" / "models" / "mnist_cnn.pt"

_MNIST = None
_MODEL = None


def _get_model():
    """Lazy: `import app.cvlib.mnist` pulls in torch (~1-3 s) and the weights
    must never load at import/registration time. First p2 call pays the cost,
    later calls reuse the module-level cache."""
    global _MNIST, _MODEL
    if _MODEL is None:
        from app.cvlib import mnist
        _MNIST = mnist
        _MODEL = mnist.load_model(WEIGHTS)
    return _MNIST, _MODEL


@register_task(
    id="p2-digit",
    name="Recognize handwritten digit",
    project=PROJECT,
    description="Classify a handwritten digit (0-9) with a CNN trained on MNIST.",
    params=[
        {
            "name": "invert",
            "type": "select",
            "options": ["auto", "always", "never"],
            "default": "auto",
            "label": "Invert polarity",
        },
        {
            "name": "pad",
            "type": "int",
            "min": 0,
            "max": 60,
            "step": 1,
            "default": 20,
            "label": "Pad around digit (px)",
            "group": "Preprocess",
        },
    ],
    output="image+data",
    result_tab="output",
    docs=DOCS_DIR / "p2-digit.md",
)
def p2_digit(img, params):
    mnist, model = _get_model()
    img28 = mnist.preprocess(
        img, invert=params.get("invert", "auto"), pad=params.get("pad", 20)
    )
    top3 = mnist.predict(model, img28)
    data = {"top3": top3}
    if not img28.any():
        data["note"] = "no digit detected"
    # show the exact tensor the model saw, enlarged for visibility
    view = cv2.resize(img28, (280, 280), interpolation=cv2.INTER_NEAREST)
    return view, data
