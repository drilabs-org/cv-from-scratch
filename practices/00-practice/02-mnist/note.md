các package

torch -> framework deep learning. Cung cấp tensor (mảng nhiều chiều + tính đạo hàm tự động), xây mạng neural (nn.Linear, nn.Conv2d...), optimizer (SGD, Adam). Đây là phần lõi để định nghĩa và train model.
torchvision -> thư viện có sẳn gồm cả dataset cho cv
scikit-learn -> thư viện machine learning "truyền thống" (không phải deep learning). Trong bài MNIST thường dùng để đánh giá model: accuracy_score, confusion_matrix, classification_report. Ngoài ra nó có sẵn các thuật toán như SVM, RandomForest, KMeans... để so sánh với neural net.

---

**Cấu trúc đầy đủ của file MNIST thường gồm 5 phần:**

1. **Load data** ✅ (bạn đã có) - dataset + DataLoader. `ToTensor` chuyển ảnh thành tensor `[0,1]`, `Normalize` đưa về quanh 0 dùng mean/std của MNIST.

2. **Định nghĩa model** - class kế thừa `nn.Module`, ví dụ đơn giản nhất: flatten 28×28=784 → `nn.Linear(784, 128)` → `nn.ReLU()` → `nn.Linear(128, 10)`. Gửi model lên device bằng `.to(device)` (device lấy từ `check_env.py`).

3. **Loss + optimizer** - `nn.CrossEntropyLoss()` cho bài toán phân loại 10 lớp; `optim.Adam(model.parameters(), lr=1e-3)` hoặc `optim.SGD`.

4. **Training loop** - lặp qua epochs, mỗi batch:
   - đưa `images, labels` lên device
   - `optimizer.zero_grad()` → `output = model(images)` → `loss = criterion(output, labels)` → `loss.backward()` → `optimizer.step()`

5. **Đánh giá** - trong `with torch.no_grad():`, đếm số dự đoán đúng trên `test_loader`, in accuracy.

Thứ tự viết: sửa bug → model → loss/optimizer → train loop → eval. Cần thêm `import torch` và `import torch.nn as nn`, `import torch.optim as optim` ở đầu file.

Bạn thử viết tiếp phần 2-5, kẹt chỗ nào hỏi lại.

---

Đây là **mean và std của toàn bộ dataset MNIST**, đã được người ta tính sẵn:

```python
transforms.Normalize((0.1307,), (0.3081,))
#                      mean      std
```

Công thức nó áp lên từng pixel:

```
pixel_mới = (pixel - 0.1307) / 0.3081
```

Trước đó `ToTensor()` đã đưa pixel từ `[0, 255]` về `[0, 1]`. Sau `Normalize` thì dữ liệu nằm quanh 0, khoảng `[-0.42, 2.82]`.

Vì sao cần: mạng neural học ổn định và hội tụ nhanh hơn khi input có mean ~0, std ~1. Nếu bỏ dòng này model vẫn train được nhưng chậm hơn/có thể kém chính xác hơn.

Vì sao là tuple `(0.1307,)` chứ không phải số trần: `Normalize` nhận mean/std **theo từng kênh màu**. MNIST là ảnh xám nên chỉ 1 kênh → tuple 1 phần tử. Với ảnh màu RGB sẽ là 3 phần tử, ví dụ `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`.

Hai con số `0.1307` và `0.3081` là kết quả thống kê trên 60,000 ảnh train MNIST - con số "kinh điển" mọi tutorial đều dùng, không phải config tự chọn. Bạn có thể tự tính lại để kiểm chứng bằng cách duyệt `train_set` và lấy `mean()/std()` của toàn bộ pixel.

---

`datasets.MNIST(download=True)` tải về 4 file dữ liệu gốc. Mỗi file có 2 bản: `.gz` (file nén tải về) và file đã giải nén (không đuôi).

| File                      | Nội dung                              |
| ------------------------- | ------------------------------------- |
| `train-images-idx3-ubyte` | 60,000 ảnh train, mỗi ảnh 28×28 pixel |
| `train-labels-idx1-ubyte` | 60,000 nhãn tương ứng (số 0-9)        |
| `t10k-images-idx3-ubyte`  | 10,000 ảnh test                       |
| `t10k-labels-idx1-ubyte`  | 10,000 nhãn test                      |

Giải mã cái tên:

- **`train`** = tập train, **`t10k`** = "test 10k" (10 nghìn ảnh test)
- **`idx1`/`idx3`** = định dạng nhị phân IDX của MNIST: idx1 là vector 1 chiều (nhãn), idx3 là mảng 3 chiều (số ảnh × 28 × 28). Đây là format gốc của dataset, không phải PNG/JPG
- **`ubyte`** = unsigned byte, mỗi pixel/nhãn là 1 byte giá trị 0-255 (ảnh grayscale)

Kiểm chứng bằng dung lượng:

- `train-images`: 47,040,016 bytes = 16 bytes header + 60,000 × 28 × 28 pixel
- `train-labels`: 60,008 bytes = 8 bytes header + 60,000 nhãn

`datasets.MNIST` khi load sẽ tự đọc đúng file train hay test dựa vào cờ `train=True/False` bạn truyền vào, nên không cần đụng vào mấy file này.

---

Q: 2 bộ này cùng load về từ 1 chỗ vậy có overlap lẫn nhau ko?
A:
Không overlap - đó chính là lý do tách thành 2 bộ.

Nhìn lại thư mục `raw` lúc nãy: `train-images-*` và `t10k-images-*` là **2 file riêng biệt**. Người tạo dataset MNIST đã chia sẵn: 60k ảnh train + 10k ảnh test, là các chữ viết tay khác nhau (thậm chí khác người viết). Cờ `train=True/False` chỉ để chọn đọc file nào, không phải tự split từ 1 file chung.

Vì sao phải tách biệt: nếu test trên chính dữ liệu đã train, model có thể chỉ "học thuộc" đáp án → accuracy ảo cao. Test set mô phỏng "dữ liệu chưa từng thấy", đo khả năng **tổng quát hóa** thật của model. Đây là nguyên tắc cơ bản nhất của machine learning.

Hai lưu ý nhỏ:

- **Cùng phân phối, khác mẫu**: 2 bộ không trùng ảnh nhưng cùng miền dữ liệu (đều chữ viết tay Mỹ), nên đánh giá vẫn công bằng.
- Thực tế MNIST có vài chục nhãn bị sai/nhòe trong dataset gốc (tài liệu QMNIST có liệt kê) nhưng tỉ lệ nhỏ, không ảnh hưởng bài học.

Ngoài ra trong thực hành người ta còn tách thêm **validation set** từ train set (vd 50k train + 10k val) để tune hyperparameter, tránh "nhìn trộm" test set. Bài này đơn giản nên 2 bộ là đủ.

---

`torch.Size([64, 1, 28, 28])` là shape của tensor batch ảnh, theo quy ước **NCHW** của PyTorch:

```
[64,  1,   28,   28 ]
 N    C    H    W
 │    │    │    └─ Width:  28 pixel chiều ngang
 │    │    └─────── Height: 28 pixel chiều dọc
 │    └──────────── Channels: 1 kênh (ảnh xám; ảnh màu RGB sẽ là 3)
 └───────────────── N (batch size): 64 ảnh, đúng batch_size=64 của DataLoader
```

Vài điểm hay gặp:

- **`torch.Size`** thực chất là tuple, `images.shape == (64, 1, 28, 28)` trả về `True`. Truy cập từng chiều: `images.shape[0]` → 64.
- **NCHW là quy ước của PyTorch**. TensorFlow/Keras mặc định NHWC (kênh nằm cuối). Đổi framework là đổi thứ tự chiều - lỗi hay gặp khi port code.
- Phần tử trong tensor lúc này là `float32` đã qua normalize (giá trị quanh 0), không còn là số nguyên 0-255. Check bằng `images.dtype`.
- `images[i, 0]` ở đoạn vẽ hình lấy ra shape `[28, 28]` - mặt nạ 1 kênh bị bỏ đi vì `imshow` chỉ cần ma trận 2D cho ảnh xám.

Model sau này nếu dùng `nn.Linear` thì phải `images.view(64, -1)` hoặc `nn.Flatten()` để duỗi `1×28×28` thành vector `784` trước khi đưa vào.
