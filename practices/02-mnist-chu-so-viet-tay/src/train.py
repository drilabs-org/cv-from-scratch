"""Train DigitCNN on MNIST; saves ../models/mnist_cnn.pt.

Standalone script following docs/02-thuc-hanh.md sections 2-5:

    cd practices/02-mnist-chu-so-viet-tay/src
    python train.py                 # 3 epochs, CPU (MNIST is small enough)
    python train.py --epochs 5 --device mps

Downloads MNIST (~60 MB) into ../data on first run. Prints per-epoch
loss/accuracy, overall test accuracy, per-digit accuracy, then saves the
state_dict that mnist.load_model() expects.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist import DigitCNN, MNIST_MEAN, MNIST_STD  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--out", type=Path, default=MODEL_DIR / "mnist_cnn.pt")
    ap.add_argument("--device", default="cpu",
                    choices=["cpu", "mps", "cuda", "auto"],
                    help="cpu is the default: MNIST trains in ~1 min and "
                         "mps crashed with torch 2.15 on macOS 14")
    args = ap.parse_args()

    device = pick_device() if args.device == "auto" else args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"PyTorch {torch.__version__} | device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train_set = datasets.MNIST(str(DATA_DIR), train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST(str(DATA_DIR), train=False, download=True,
                              transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    print(f"train: {len(train_set)} imgs | test: {len(test_set)} imgs")

    model = DigitCNN().to(device)
    print(f"DigitCNN params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"loss {total_loss / seen:.4f} | "
              f"train acc {correct / seen:.4f}")

    model.eval()
    correct, seen = 0, 0
    class_correct = [0] * 10
    class_total = [0] * 10
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images.to(device)).argmax(1).cpu()
            correct += (preds == labels).sum().item()
            seen += labels.size(0)
            for p, lbl in zip(preds.tolist(), labels.tolist()):
                class_total[lbl] += 1
                class_correct[lbl] += int(p == lbl)
    print(f"Test acc: {correct / seen:.4f} ({correct}/{seen})")
    for d in range(10):
        print(f"  digit {d}: {class_correct[d] / class_total[d]:.4f} "
              f"({class_total[d]} samples)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
