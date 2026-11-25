"""Tasks for project 2 (mnist-digits): MNIST digit CNN."""
import os
import pickle
from pathlib import Path

import cv2

from app.registry import TaskExecutionError, register_task

PROJECT = "mnist-digits"
DOCS_DIR = Path(__file__).resolve().parent / "docs"
BUNDLED_WEIGHTS = (
    Path(__file__).resolve().parents[1] / "cvlib" / "models" / "mnist_cnn.pt"
)
MODEL_FILENAME = "mnist_cnn.pt"

_MNIST = None
_MODEL = None
_MODEL_RUNTIME_INFO = None


def _resolve_model_weights():
    models_dir = os.environ.get("CV_MODELS_DIR", "").strip()
    if not models_dir:
        return BUNDLED_WEIGHTS, {
            "source": "bundled",
            "fallback": False,
            "message": "Using the bundled MNIST model (CV_MODELS_DIR is not set).",
        }

    custom_weights = Path(models_dir).expanduser() / MODEL_FILENAME
    if custom_weights.is_file():
        return custom_weights, {
            "source": "custom",
            "fallback": False,
            "message": f"Using custom MNIST weights from {custom_weights}.",
        }

    return BUNDLED_WEIGHTS, {
        "source": "bundled",
        "fallback": True,
        "message": (
            f"Custom MNIST weights were not found at {custom_weights}; "
            "using the bundled default model."
        ),
    }


def _get_model():
    """Lazy: `import app.cvlib.mnist` pulls in torch (~1-3 s) and the weights
    must never load at import/registration time. First p2 call pays the cost,
    later calls reuse the module-level cache."""
    global _MNIST, _MODEL, _MODEL_RUNTIME_INFO
    if _MODEL is None:
        from app.cvlib import mnist

        weights_path, runtime_info = _resolve_model_weights()
        try:
            model = mnist.load_model(weights_path)
        except (
            OSError,
            RuntimeError,
            EOFError,
            ValueError,
            TypeError,
            pickle.UnpicklingError,
        ) as e:
            raise TaskExecutionError(
                f"Failed to load MNIST weights from {weights_path}: {e}. "
                "The file must be a DigitCNN state_dict with the expected "
                "MNIST preprocessing and 10 output classes."
            ) from e

        _MNIST = mnist
        _MODEL = model
        _MODEL_RUNTIME_INFO = runtime_info
    return _MNIST, _MODEL


def _runtime_info():
    return _MODEL_RUNTIME_INFO


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
    runtime_info=_runtime_info,
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
