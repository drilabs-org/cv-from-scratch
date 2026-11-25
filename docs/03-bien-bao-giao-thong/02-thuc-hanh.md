---
title: "Thực hành: Nhận diện biển báo giao thông"
description: "Thực hành phân loại 43 loại biển báo trên GTSRB: EDA, augmentation, transfer learning ResNet18, đánh giá theo lớp và đăng ký task p3-* trong CV Playground."
---

# Thực hành: Nhận diện biển báo giao thông

Làm theo thứ tự từ trên xuống. Khái niệm chi tiết xem [Lý thuyết](./01-ly-thuyet).

## 1. Cài đặt và tải dữ liệu

Tiếp tục trong **CV Playground**: module model viết trong `backend/app/cvlib/gtsrb.py`, task đăng ký trong `backend/app/tasks/p3_gtsrb.py`. Venv đã có `torch`, `torchvision`, `pillow`, chỉ cần thêm package phân tích:

```bash
cd cv-playground
source .venv/bin/activate    # bỏ qua nếu chưa tạo: xem mục cài đặt của Project 1
pip install -r requirements.txt
pip install matplotlib scikit-learn
```

Các đoạn code dưới đây giả định chạy từ `backend/app/cvlib`, để dataset rơi vào `cvlib/data` và weights vào `cvlib/models` đúng convention.

Tải GTSRB qua torchvision (~260MB lần đầu):

```python
from torchvision import datasets

train_set = datasets.GTSRB("./data", split="train", download=True)
test_set = datasets.GTSRB("./data", split="test", download=True)

print(len(train_set), len(test_set))   # 26640 12630

img, label = train_set[0]
print(type(img), img.size, label)      # PIL.Image, kich thuoc moi anh khac nhau
```

> [!NOTE]
> Tham số là `split=` chứ không `train=` như MNIST. Dataset lưu trong `data/gtsrb/`, tổ chức thành 43 thư mục theo lớp, kèm `Meta.csv` chứa tên đầy đủ các lớp.

Bảng tên lớp để map `label -> tên` khi predict (điền đủ từ `Meta.csv`):

```python
CLASS_NAMES = {
    0: "Gioi han 20", 1: "Gioi han 30", 2: "Gioi han 50",
    3: "Gioi han 60", 4: "Gioi han 70", 5: "Gioi han 80",
    13: "Nhuong duong", 14: "Dung lai", 17: "Cam di vao",
    # ... dien du 43 lop tu Meta.csv
}
```

## 2. Khám phá dữ liệu (EDA)

Phân bố số mẫu theo lớp:

```python
from collections import Counter
import matplotlib.pyplot as plt

counts = Counter(label for _, label in train_set)
ids = sorted(counts)
plt.figure(figsize=(14, 4))
plt.bar(ids, [counts[i] for i in ids])
plt.xlabel("Class ID"); plt.ylabel("So mau")
plt.show()

print("Mat can bang:", max(counts.values()) / min(counts.values()))
```

Kích thước ảnh:

```python
sizes = [img.size for img, _ in list(train_set)[:2000]]
ws, hs = zip(*sizes)
print(f"W: {min(ws)}-{max(ws)}, H: {min(hs)}-{max(hs)}")
```

Nhìn mẫu ngẫu nhiên:

```python
import random

fig, axes = plt.subplots(5, 8, figsize=(14, 8))
for i, ax in enumerate(axes.flat):
    img, label = train_set[random.randrange(len(train_set))]
    ax.imshow(img)
    ax.set_title(str(label), fontsize=8)
    ax.axis("off")
plt.show()
```

## 3. Pipeline augmentation

```python
from torchvision import transforms

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.3),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    normalize,
])
```

Xem augmentation hoạt động (cùng một ảnh, nhiều phiên bản):

```python
aug_set = datasets.GTSRB("./data", split="train",
                         transform=train_transform)
idx = 100
fig, axes = plt.subplots(1, 6, figsize=(12, 2))
axes[0].imshow(datasets.GTSRB("./data", split="train")[idx][0])
for i in range(1, 6):
    t = aug_set[idx][0].permute(1, 2, 0) * 0.5 + 0.5
    axes[i].imshow(t.clamp(0, 1))
for ax in axes: ax.axis("off")
plt.show()
```

DataLoader, kèm sampler bù mất cân bằng (chọn 1 trong 2 cách):

```python
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

train_set = datasets.GTSRB("./data", split="train",
                           transform=train_transform)
test_set = datasets.GTSRB("./data", split="test",
                          transform=test_transform)

# Cach 1: sampler (khong dung shuffle khi da co sampler)
labels = [l for _, l in train_set]
weights = 1.0 / np.bincount(labels)[labels]
sampler = WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(train_set, batch_size=128,
                          sampler=sampler, num_workers=2)
test_loader = DataLoader(test_set, batch_size=256,
                         shuffle=False, num_workers=2)
```

## 4. Transfer learning với ResNet18

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights="IMAGENET1K_V1")

# Chien luoc 1: dong bang backbone, chi train fc
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 43)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

Sau vài epoch, nếu chưa đạt thì mở `layer4` fine-tune:

```python
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])
```

## 5. Huấn luyện

Training loop tái sử dụng từ Project 2:

```python
device = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
criterion = nn.CrossEntropyLoss()

num_epochs = 5
for epoch in range(num_epochs):
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

    print(f"Epoch {epoch + 1}/{num_epochs} | "
          f"loss {total_loss / seen:.4f} | train acc {correct / seen:.4f}")

torch.save(model.state_dict(), "models/gtsrb_resnet18.pt")
```

## 6. Đánh giá

Test accuracy tổng:

```python
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images.to(device)).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

from sklearn.metrics import accuracy_score
print("Test acc:", accuracy_score(all_labels, all_preds))
```

Per-class accuracy, tìm lớp yếu:

```python
all_preds, all_labels = np.array(all_preds), np.array(all_labels)
for c in range(43):
    mask = all_labels == c
    print(f"Lop {c:2d} ({mask.sum():4d} mau): "
          f"{(all_preds[mask] == c).mean():.3f}")
```

Confusion matrix cho cụm giới hạn tốc độ (id 0-8):

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

speed_ids = list(range(9))
mask = np.isin(all_labels, speed_ids)
cm = confusion_matrix(all_labels[mask], all_preds[mask],
                      labels=speed_ids)
ConfusionMatrixDisplay(cm, display_labels=speed_ids).plot(cmap="Blues")
plt.show()
```

## 7. Module gtsrb và task `p3-sign`

Logic model gom trong `backend/app/cvlib/gtsrb.py`, theo đúng pattern của `mnist.py` ở Project 2 (định nghĩa + inference một chỗ):

```python
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CLASS_NAMES = {
    0: "Gioi han 20", 1: "Gioi han 30", 2: "Gioi han 50",
    # ... dien du 43 lop
}

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_model(weights_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 43)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, img_bgr):
    """BGR ndarray -> top-3 [{'label': ..., 'prob': ...}]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = transform(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = probs.topk(3)
    return [{"label": CLASS_NAMES.get(int(i), f"lop {i}"),
             "prob": round(float(v), 4)}
            for v, i in zip(top.values, top.indices)]
```

File đăng ký `backend/app/tasks/p3_gtsrb.py` gần như copy khung của `p2_mnist.py`: nạp model lazy, trả `output="data"` vì task chỉ trả nhãn, không có ảnh đầu ra:

```python
from pathlib import Path

from app.registry import register_task

PROJECT = "traffic-signs"
DOCS_DIR = Path(__file__).resolve().parent / "docs"
WEIGHTS = (Path(__file__).resolve().parents[1]
           / "cvlib" / "models" / "gtsrb_resnet18.pt")

_GTSRB = None
_MODEL = None


def _get_model():
    """Lazy: import torch + nap weights ton vai giay, khong duoc lam
    luc module dang duoc import de dang ky task."""
    global _GTSRB, _MODEL
    if _MODEL is None:
        from app.cvlib import gtsrb
        _GTSRB = gtsrb
        _MODEL = gtsrb.load_model(WEIGHTS)
    return _GTSRB, _MODEL


@register_task(
    id="p3-sign",
    name="Traffic sign recognition",
    project=PROJECT,
    description="Classify a traffic sign into 43 GTSRB classes "
                "with fine-tuned ResNet18.",
    output="data",
    docs=DOCS_DIR / "p3-sign.md",
)
def p3_sign(img, params):
    gtsrb, model = _get_model()
    return None, {"top3": gtsrb.predict(model, img)}
```

Đừng quên hai thứ nhỏ nhưng bắt buộc: thêm `from app.tasks import p3_gtsrb` vào `backend/app/main.py` để task được đăng ký, và tạo `tasks/docs/p3-sign.md` nếu muốn có panel tài liệu.

Chạy `./dev.sh`, upload ảnh biển báo vào `p3-sign`, kết quả mong đợi:

```text
top3: Dung lai (94.20%) | Nhuong duong (3.80%) | Cam di vao (1.10%)
```

> [!WARNING]
> Transform lúc predict phải y hệt `test_transform` lúc train. Sai normalize là lỗi số 1 khiến model tốt lúc test nhưng tệ lúc dùng thật.

> [!NOTE]
> Task trả ảnh `None` là hợp lệ: `output="data"` báo cho frontend chỉ hiển thị phần số liệu. Muốn vẽ nhãn lên ảnh và trả về thì đổi thành `output="image+data"`.

## Bài tập

1. Thử thêm `transforms.RandomPerspective(0.2)`, quyết định có giữ không và vì sao.
2. So sánh 3 cấu hình: không augmentation, augmentation, augmentation + fine-tune `layer4`.
3. Tìm 10 ảnh test model đoán sai với độ tin cậy cao, phân loại lỗi.
4. Tăng input lên 96x96 hoặc 128x128, xem cụm tốc độ có cải thiện không.
5. Thay `resnet18` bằng `mobilenet_v3_small`, so sánh tốc độ và accuracy.

## Checklist tự đánh giá

- [ ] Vì sao dataset thật cần EDA trước khi model.
- [ ] Chọn augmentation không phá nhãn, vì sao không lật ngang biển báo.
- [ ] Feature extraction vs fine-tune, khi nào dùng cái nào.
- [ ] Đọc per-class accuracy và confusion matrix tìm lớp yếu.
- [ ] Classification khác detection thế nào.

## Hướng mở rộng

- Dùng YOLO tìm biển báo trong ảnh cảnh đường toàn cảnh (detection), rồi đăng ký thêm task `p5-*` trong CV Playground.
- Thêm task variant của `p3-sign` hiển thị Grad-CAM: model trả vùng "nhìn vào đâu" thay vì chỉ top-3.
- Thử pipeline trên dataset tự thu (ví dụ 5 loại trái cây), đăng ký thành task mới cùng `project` tag riêng.

## Hoàn thành

- [Quay lại lộ trình](../)
