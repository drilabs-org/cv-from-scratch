# ============================================================
# MNIST - Nhan dang chu so viet tay bang CNN
# Pipeline: load data -> dinh nghia model -> train -> danh gia
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

# Pipeline xu ly anh truoc khi dua vao model:
#   ToTensor  : PIL Image -> Tensor, scale pixel [0,255] -> [0,1]
#   Normalize : chuan hoa ve quanh 0 bang mean/std cua MNIST
#               (anh xam nen chi co 1 kenh -> tuple 1 phan tu)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Dataset: tu tai ve ./data neu chua co (download=True)
#   train=True  -> doc train-images/labels (60k anh)
#   train=False -> doc t10k-images/labels  (10k anh test)
train_set = datasets.MNIST("./data", train=True, download=True,
                           transform=transform)
test_set = datasets.MNIST("./data", train=False, download=True,
                          transform=transform)

# DataLoader: chia dataset thanh tung batch de train
#   train: batch 64, xao tron moi epoch (model khong hoc thuoc thu tu)
#   test : batch 1000, khong can xao
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)


# ------------------------------------------------------------
# 2. DINH NGHIA MODEL (CNN)
# ------------------------------------------------------------
# Input: [N, 1, 28, 28]  (batch, kenh, cao, rong)
# Output: [N, 10]        (10 logit ung voi chu so 0-9)
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Phan trich dac trung: conv + relu + pool lap lai 2 lan
        # Conv giau kenh, Pool giam kich thuoc khong gian
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),    # 1x28x28 -> 32x26x26
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> 32x13x13
            nn.Conv2d(32, 64, kernel_size=3),   # -> 64x11x11
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> 64x5x5
        )

        # Phan phan loai: duoi phang roi qua 2 lop Linear
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # -> 64*5*5 = 1600
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),                 # -> 10 logit
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ------------------------------------------------------------
# SANITY CHECK
# ------------------------------------------------------------
model = DigitCNN()
# print(model)    # in kien truc mang + shape tung lop
x = torch.randn(1, 1, 28, 28)
print(model(x).shape)                             # [1, 10]
print(sum(p.numel() for p in model.parameters())) # ~421k

# ------------------------------------------------------------
# XEM NHANH 1 BATCH (mo lai khi can kiem tra data)
# ------------------------------------------------------------
# images, labels = next(iter(train_loader))
# print(images.shape)   # [64, 1, 28, 28]
#
# fig, axes = plt.subplots(2, 8, figsize=(12, 3))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(images[i, 0], cmap="gray")
#     ax.set_title(int(labels[i]))
#     ax.axis("off")
# plt.tight_layout()
# plt.show()


# ------------------------------------------------------------
# TODO: PHAN TIEP THEO
# ------------------------------------------------------------
# 3. device (mps/cuda/cpu) + model.to(device)
#    loss = nn.CrossEntropyLoss()
#    optimizer = optim.Adam(model.parameters(), lr=1e-3)
# 4. Training loop: for epoch -> for images, labels in train_loader
#    -> zero_grad / forward / loss / backward / step
# 5. Eval: model.eval() + torch.no_grad() -> dem so du doan dung
#    tren test_loader -> in accuracy
