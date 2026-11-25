"""MNIST digit recognition: model definition + inference helpers.

Everything inference needs lives here so both the training script and the
app task handlers can import one module. Training lives in train.py.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class DigitCNN(nn.Module):
    """Must match the keys saved in models/mnist_cnn.pt."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # 1x28x28 -> 32x26x26
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> 32x13x13
            nn.Conv2d(32, 64, kernel_size=3),  # -> 64x11x11
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> 64x5x5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # -> 1600
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def preprocess(img, invert="auto", pad=20):
    """BGR ndarray -> 28x28 uint8 matching MNIST (white digit on black).

    The model only knows white-strokes-on-black, so a photo of ink on paper
    must be polarity-fixed, cropped to the digit, padded square, then resized.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if invert == "always" or (invert == "auto" and (binary > 0).mean() > 0.5):
        binary = cv2.bitwise_not(binary)
    coords = cv2.findNonZero(binary)
    if coords is None:                       # blank input: degrade, don't crash
        return cv2.resize(binary, (28, 28))
    x, y, w, h = cv2.boundingRect(coords)
    digit = binary[y:y + h, x:x + w]
    side = max(w, h) + pad
    canvas = np.zeros((side, side), np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h,
           (side - w) // 2:(side - w) // 2 + w] = digit
    return cv2.resize(canvas, (28, 28))


def load_model(weights_path):
    """CPU only: the model is tiny and CPU inference is deterministic."""
    model = DigitCNN()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, img28):
    """28x28 uint8 -> top-3 [{"label": "7", "prob": 0.98}, ...]."""
    x = torch.tensor(img28, dtype=torch.float32).div(255)
    x = (x - MNIST_MEAN) / MNIST_STD
    with torch.no_grad():
        probs = torch.softmax(model(x[None, None]), dim=1)[0]
    top = probs.topk(3)
    return [{"label": str(int(i)), "prob": round(float(p), 4)}
            for p, i in zip(top.values, top.indices)]
