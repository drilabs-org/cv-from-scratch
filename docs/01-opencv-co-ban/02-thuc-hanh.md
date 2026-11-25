---
title: "Thực hành: Xử lý ảnh với OpenCV"
description: "Thực hành xử lý ảnh với OpenCV từ cài đặt đến mini app imgtool hoàn chỉnh: đọc ảnh, lọc màu, threshold, tìm contour và đếm vật thể."
---

# Thực hành: Xử lý ảnh với OpenCV

Làm theo thứ tự từ trên xuống. Khái niệm chi tiết xem [Lý thuyết](./01-ly-thuyet); nếu chưa quen NumPy, đọc trước [Python cho Computer Vision](./00-python-for-cv).

## 1. Cài đặt môi trường

```bash
mkdir cv-opencv-basics
cd cv-opencv-basics
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy matplotlib
```

Trong đó: `opencv-python` là thư viện chính (import bằng `cv2`), `numpy` vì ảnh trong OpenCV chính là mảng NumPy, `matplotlib` để hiển thị ảnh (`cv2.imshow` hay crash trên macOS nên tài liệu này dùng `plt.imshow`).

### Tạo ảnh mẫu

Lưu đoạn này thành `make_samples.py` trong `src/` rồi chạy:

```python
import cv2
import numpy as np

img = np.full((400, 400, 3), 255, dtype=np.uint8)
cv2.rectangle(img, (30, 30), (150, 150), (200, 60, 40), -1)
cv2.circle(img, (300, 100), 60, (60, 180, 75), -1)
cv2.putText(img, "CV", (120, 320), cv2.FONT_HERSHEY_SIMPLEX,
            3, (30, 30, 30), 8)
cv2.imwrite("sample.png", img)
```

### Kiểm tra cài đặt

```python
import cv2

img = cv2.imread("sample.png")
print("Shape:", img.shape)   # (400, 400, 3)
print("Dtype:", img.dtype)   # uint8
```

> [!WARNING]
> `cv2.imread` trả về `None` (không báo lỗi) khi đường dẫn sai. Luôn kiểm tra `img is not None` trước khi xử lý.

## 2. Thao tác cơ bản trên ảnh

```python
img = cv2.imread("sample.png")

print(img[0, 0])            # pixel góc trên-trái: [B G R]
print(img[100, 50])         # img[y, x]: hàng 100, cột 50

roi = img[30:150, 30:150]   # cat vung bang slicing img[y1:y2, x1:x2]
small = cv2.resize(img, (200, 200))          # (width, height)
half = cv2.resize(img, None, fx=0.5, fy=0.5) # theo ti le

img[:, :, 2] = 0            # tat kenh do
```

Hiển thị đúng màu bằng matplotlib:

```python
import matplotlib.pyplot as plt

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb); plt.show()
```

> [!WARNING]
> Hai quy ước ngược nhau dễ gây bug: indexing là `img[y, x]` nhưng `cv2.resize` nhận `(width, height)`, còn `shape` lại là `(height, width)`.

## 3. Không gian màu và histogram

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # (400, 400) 1 kenh
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Loc vung mau do (hue quanh 0 nen can 2 khoang)
mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 80, 80), (179, 255, 255))
mask = mask1 | mask2
result = cv2.bitwise_and(img, img, mask=mask)

# Histogram va can bang
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(hist); plt.show()
gray_eq = cv2.equalizeHist(gray)
```

## 4. Lọc ảnh và ngưỡng

```python
import numpy as np

# Kernel tu dinh nghia
kernel = np.ones((3, 3), np.float32) / 9
blurred = cv2.filter2D(img, -1, kernel)

# 3 phep mo co san (kernel size phai le)
gauss = cv2.GaussianBlur(gray, (5, 5), 0)
median = cv2.medianBlur(gray, 5)

# Lam net
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], np.float32)
sharp = cv2.filter2D(img, -1, kernel_sharpen)

# 3 cach threshold
_, b1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, b2 = cv2.threshold(gray, 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
b3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY, 11, 2)
```

Pipeline tiêu chuẩn để chuẩn bị ảnh nhị phân:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)     # khu nhieu truoc
_, binary = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

## 5. Biên và contour

```python
edges = cv2.Canny(blur, 50, 150)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)

out = img.copy()
for c in contours:
    area = cv2.contourArea(c)
    if area < 500:                  # bo nhieu nho
        continue
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    # len(approx) == 3: tam giac | == 4: tu giac | > 4: gan tron
```

Đếm đồng xu (đặt xu trên nền tối, chụp thẳng):

```python
img = cv2.imread("coins.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
_, binary = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
coins = [c for c in contours if cv2.contourArea(c) > 1000]
print("So dong xu:", len(coins))
```

## 6. Mini app imgtool

Gộp pipeline thành CLI. Lưu thành `src/imgtool.py`:

```python
import argparse
import cv2


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_blur(img):
    return cv2.GaussianBlur(to_gray(img), (5, 5), 0)


def to_edges(img):
    return cv2.Canny(to_blur(img), 50, 150)


def to_binary(img):
    _, b = cv2.threshold(to_blur(img), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def count_objects(img):
    contours, _ = cv2.findContours(to_binary(img), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    kept = [c for c in contours if cv2.contourArea(c) > 500]
    for c in kept:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out, len(kept)


def main():
    parser = argparse.ArgumentParser(description="Cong cu xu ly anh don gian")
    parser.add_argument("image", help="Duong dan anh dau vao")
    parser.add_argument("command",
                        choices=["gray", "blur", "edges", "binary", "count"])
    parser.add_argument("--out", default="output.png")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Khong doc duoc anh: {args.image}")

    if args.command == "count":
        out, n = count_objects(img)
        print(f"Tim thay {n} vat the")
    else:
        out = {
            "gray": to_gray,
            "blur": to_blur,
            "edges": to_edges,
            "binary": to_binary,
        }[args.command](img)

    cv2.imwrite(args.out, out)
    print(f"Da luu: {args.out}")


if __name__ == "__main__":
    main()
```

Chạy thử:

```bash
python src/imgtool.py sample.png gray
python src/imgtool.py sample.png edges --out edges.png
python src/imgtool.py coins.jpg count --out counted.png
```

Điểm thiết kế đáng chú ý: hàm xếp tầng (`to_binary` gọi `to_blur` gọi `to_gray`), dict dispatch thay chuỗi if/elif, fail fast ngay sau `imread`.

## Bài tập

1. Thêm hình tam giác vào `make_samples.py` bằng `cv2.polylines`.
2. Chụp ảnh có vật màu nổi bật, dùng HSV + `inRange` tách riêng vật đó.
3. Chụp trang giấy dưới ánh sáng không đều, so sánh 3 cách threshold.
4. Đếm số phím trên ảnh bàn phím chụp, nhận xét khi nào pipeline thất bại.
5. Thêm lệnh `hsv-mask` cho imgtool nhận tham số khoảng màu.
6. Thêm lệnh `scan` mô phỏng scanner: tìm contour 4 đỉnh lớn nhất rồi `cv2.warpPerspective` nắn thẳng.

## Checklist tự đánh giá

- [ ] Ảnh màu `(H, W, 3)` trong OpenCV theo thứ tự kênh nào?
- [ ] `img[y, x]` và `cv2.resize(img, (w, h))` khác nhau thế nào?
- [ ] Khi nào chọn HSV thay RGB để lọc màu?
- [ ] Vì sao blur trước khi Canny hoặc threshold?
- [ ] Kể được pipeline đếm vật thể từ ảnh màu.

## Tiếp theo

- [Project 2: Nhận diện chữ số viết tay](../02-mnist-chu-so-viet-tay/01-ly-thuyet)
