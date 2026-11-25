---
title: "Thực hành: Xử lý ảnh với OpenCV"
description: "Thực hành xử lý ảnh với OpenCV từ cài đặt đến các task p1-* trên CV Playground: đọc ảnh, lọc màu, threshold, tìm contour và đếm vật thể."
---

# Thực hành: Xử lý ảnh với OpenCV

Làm theo thứ tự từ trên xuống. Khái niệm chi tiết xem [Lý thuyết](./01-ly-thuyet); nếu chưa quen NumPy, đọc trước [Python cho Computer Vision](./00-python-for-cv).

## 1. Cài đặt môi trường

Sản phẩm của project là các task trong **CV Playground**, web app có sẵn trong repo. Code bạn viết nằm ở hai chỗ: logic xử lý trong `backend/app/cvlib/`, đăng ký task trong `backend/app/tasks/`. Dựng môi trường và chạy app:

```bash
cd cv-playground
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install matplotlib       # chỉ để hiển thị ảnh trong các đoạn thử nghiệm dưới đây
./dev.sh                     # backend :8000 + frontend :5173
```

Mở http://localhost:5173, upload một ảnh bất kỳ rồi chạy thử task `p1-grayscale` để thấy luồng hoạt động trước khi code. Các đoạn code thử nghiệm ở mục 2-5 chạy trong venv này, đặt file ở đâu cũng được (ví dụ tự tạo thư mục `scratch/`).

Trong `requirements.txt`: `opencv-python-headless` là thư viện chính (import bằng `cv2`, bản headless không kéo GUI deps), `numpy` vì ảnh trong OpenCV chính là mảng NumPy, `fastapi`/`uvicorn` chạy backend, `torch` dùng cho Project 2. `matplotlib` không có sẵn vì app hiển thị ảnh trên trình duyệt; bạn vừa cài nó để các đoạn script rời xem được ảnh (`cv2.imshow` hay crash trên macOS nên tài liệu này dùng `plt.imshow`).

### Tạo ảnh mẫu

Repo có sẵn `backend/app/cvlib/make_samples.py` sinh `sample.png` và `coins.jpg` vào thư mục đang đứng. Chạy thử:

```bash
python backend/app/cvlib/make_samples.py
```

Nội dung chính của script, đáng gõ lại một lần để quen API vẽ hình:

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

## 6. Module imgtool và các task trên web app

Gộp pipeline thành module `backend/app/cvlib/imgtool.py`. Bản trong repo đã mở tham số cho từng phép để task trên web chỉnh được:

```python
import argparse
import cv2


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_blur(img, ksize=5, sigma=0):
    return cv2.GaussianBlur(to_gray(img), (ksize, ksize), sigma)


def to_edges(img, t1=50, t2=150, ksize=5, aperture=3, l2=False):
    return cv2.Canny(to_blur(img, ksize), t1, t2,
                     apertureSize=aperture, L2gradient=l2)


def to_binary(img, method="otsu", ksize=5, block=11, C=2):
    if method == "adaptive":
        return cv2.adaptiveThreshold(to_blur(img, ksize), 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block, C)
    _, b = cv2.threshold(to_blur(img, ksize), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def count_objects(img, min_area=500, ksize=5, method="otsu",
                  invert="auto", dilate=0, block=11, C=2):
    binary = to_binary(img, method, ksize, block, C)
    white_ratio = (binary > 0).mean()
    if invert == "always" or (invert == "auto" and white_ratio > 0.5):
        binary = cv2.bitwise_not(binary)
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
        binary = cv2.dilate(binary, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    kept = [c for c in contours if cv2.contourArea(c) > min_area]
    for c in kept:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out, len(kept)
```

(Cuối file còn `main()` dùng `argparse` để chạy thử bằng dòng lệnh, xem mục 7 của [Python cho CV](./00-python-for-cv).)

Điểm thiết kế đáng chú ý: hàm xếp tầng (`to_binary` gọi `to_blur` gọi `to_gray`), mọi hàm cùng kiểu `(img) -> img` nên lắp ghép được vào pipeline, và `count_objects` trả `(ảnh, số)` vì đầu ra gồm cả dữ liệu lẫn ảnh.

### Đăng ký thành task

File `backend/app/tasks/p1_opencv.py` bọc mỗi hàm thành một task bằng `@register_task`. Hai task tiêu biểu:

```python
from pathlib import Path

from app.cvlib import imgtool
from app.registry import register_task

PROJECT = "opencv-basics"
DOCS_DIR = Path(__file__).resolve().parent / "docs"


@register_task(
    id="p1-edges",
    name="Canny edge detection",
    project=PROJECT,
    description="Find edges in the image using the Canny algorithm.",
    params=[
        {"name": "t1", "type": "int", "min": 0, "max": 255,
         "default": 50, "label": "Low threshold"},
        {"name": "t2", "type": "int", "min": 0, "max": 255,
         "default": 150, "label": "High threshold"},
        {"name": "ksize", "type": "int", "min": 1, "max": 31,
         "step": 2, "default": 5, "label": "Blur kernel (odd)",
         "group": "Pre-blur"},
    ],
    output="image",
    docs=DOCS_DIR / "p1-edges.md",
)
def p1_edges(img, params):
    return imgtool.to_edges(img, params.get("t1", 50),
                            params.get("t2", 150),
                            ksize=params.get("ksize", 5)), {}


@register_task(
    id="p1-count",
    name="Count objects",
    project=PROJECT,
    description="Count objects via contours, drawing bounding boxes.",
    params=[
        {"name": "min_area", "type": "int", "min": 0, "max": 20000,
         "default": 500, "label": "Minimum area (px)"},
    ],
    output="image+data",
    result_tab="output",
    docs=DOCS_DIR / "p1-count.md",
)
def p1_count(img, params):
    out, n = imgtool.count_objects(img,
                                   min_area=params.get("min_area", 500))
    return out, {"count": n}
```

Đọc nhanh:

- **Chữ ký cố định `(img, params) -> (ảnh, data)`:** `img` là `ndarray` BGR từ file upload, `params` là dict đã validate theo metadata. Task trả số liệu thì dùng `output="image+data"` và dict phẳng như `{"count": n}`.
- **`params` tự sinh form:** `type` quyết định control (`int`/`float` thành slider, `select` thành dropdown), `group` xếp param vào nhóm "Advanced" gấp lại.
- **`docs` trỏ tới file markdown** trong `tasks/docs/`, hiển thị thành panel tài liệu cạnh task; thêm file `<id>.vi.md` là có tab tiếng Việt.
- **Không cần đụng frontend:** frontend đọc `GET /api/tasks` rồi tự dựng UI. Chỉ cần import module task trong `backend/app/main.py` để nó được đăng ký.

### Chạy và kiểm tra

```bash
./dev.sh     # rồi mở http://localhost:5173
```

Upload `sample.png`, chọn task `p1-edges` và kéo slider `t1`/`t2` để thấy biên thay đổi trực tiếp. Upload `coins.jpg` cho `p1-count`: kết quả gồm ảnh vẽ bbox lẫn con số `count` trên badge.

Module vẫn chạy được độc lập bằng CLI để debug nhanh:

```bash
python backend/app/cvlib/imgtool.py sample.png edges --out edges.png
python backend/app/cvlib/imgtool.py coins.jpg count --out counted.png
```

## Bài tập

1. Thêm hình tam giác vào `make_samples.py` bằng `cv2.polylines`.
2. Chụp ảnh có vật màu nổi bật, dùng HSV + `inRange` tách riêng vật đó.
3. Chụp trang giấy dưới ánh sáng không đều, so sánh `otsu` vs `adaptive` ngay trong task `p1-binary`.
4. Đếm số phím trên ảnh bàn phím chụp bằng `p1-count`, nhận xét khi nào pipeline thất bại và `dilate`/`invert` có giúp không.
5. Thêm task `p1-hsv-mask` nhận khoảng màu (gợi ý: tham số `h_min`, `h_max` kiểu `int`, trả ảnh mask).
6. Thêm task `p1-scan` mô phỏng scanner: tìm contour 4 đỉnh lớn nhất rồi `cv2.warpPerspective` nắn thẳng. Đây là bản rút gọn của Project 6.

## Checklist tự đánh giá

- [ ] Ảnh màu `(H, W, 3)` trong OpenCV theo thứ tự kênh nào?
- [ ] `img[y, x]` và `cv2.resize(img, (w, h))` khác nhau thế nào?
- [ ] Khi nào chọn HSV thay RGB để lọc màu?
- [ ] Vì sao blur trước khi Canny hoặc threshold?
- [ ] Kể được pipeline đếm vật thể từ ảnh màu.
- [ ] Đăng ký được một task mới vào CV Playground: handler `(img, params)`, metadata params, import trong `main.py`.

## Tiếp theo

- [Project 2: Nhận diện chữ số viết tay](../02-mnist-chu-so-viet-tay/01-ly-thuyet)
