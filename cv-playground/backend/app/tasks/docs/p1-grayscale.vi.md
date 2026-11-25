# Chuyển sang ảnh xám (`p1-grayscale`)

Giảm ảnh màu 3 kênh xuống còn 1 kênh chỉ giữ độ sáng. Đây là bước mở đầu của hầu hết mọi pipeline xử lý ảnh truyền thống: `to_blur`, `to_edges`, `to_binary` và `count_objects` trong `imgtool` đều gọi `to_gray` bên trong trước tiên.

## Lý thuyết: độ sáng là gì

Mỗi pixel của ảnh màu là một bộ ba $(B, G, R)$. Ảnh xám nén 3 con số đó thành một độ chói (luminance) duy nhất bằng các trọng số khớp với cảm nhận của mắt người:

$$Y = 0.299\,R + 0.587\,G + 0.114\,B \qquad \text{(ITU-R BT.601)}$$

Kênh xanh lá nặng nhất vì mắt nhạy với nó nhất, xanh dương nhẹ nhất. Trung bình thường $(R+G+B)/3$ tạo ra ảnh "phẳng", mất cảm giác sáng tự nhiên.

Ba lý do ảnh xám luôn đứng đầu pipeline:

1. **Ít dữ liệu hơn 3 lần**: $(H, W, 3) \to (H, W)$, nên mọi phép toán sau đó nhanh và đơn giản hơn.
2. **Nhiều thuật toán chỉ định nghĩa trên 1 kênh**: Canny, threshold và contours đều làm việc trên cường độ sáng.
3. **Loại bỏ nhiễu màu**: hai pixel cùng độ sáng nhưng khác hue không nên bị coi là "khác nhau" khi tìm cạnh.

Đánh đổi: thông tin màu mất vĩnh viễn. Một vùng đỏ tươi và một vùng xanh lá có cùng luminance sẽ ra cùng một mức xám và không thể tách lại được. Bài toán cần màu (lọc vật thể theo màu, chroma key) phải đi qua HSV + `inRange`, không phải grayscale.

## Tech stack: NumPy + OpenCV

- Ảnh trong OpenCV là `numpy.ndarray` với `dtype=uint8` (0..255). Ảnh xám có `shape = (H, W)`; ảnh màu có `shape = (H, W, 3)`.
- Truy cập pixel bằng `img[y, x]` (hàng trước, cột sau), ngược với trực giác toán học $(x, y)$.
- Toàn bộ việc chuyển đổi chỉ là một hàm:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

- Hằng số là `COLOR_BGR2GRAY` vì OpenCV đọc ảnh theo thứ tự kênh **BGR** (lý do lịch sử: phần cứng thời Windows khi OpenCV ra đời). Dùng nhầm `COLOR_RGB2GRAY` sẽ hoán đổi trọng số đỏ/xanh dương và tạo ảnh xám sai tông.
- `cvtColor` dùng số học fixed-point, nên tự nhân trọng số bằng NumPy float sẽ lệch vài đơn vị; không ảnh hưởng kết quả.
- Ảnh 1 kênh lưu PNG bình thường, nhưng **không vẽ được màu lên nó**: để annotate (bbox trong `p1-count`) phải chuyển lại bằng `COLOR_GRAY2BGR`.

## Code

```python
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

Task `p1-grayscale` chỉ là wrapper gọi hàm này, không có tham số.

## Mẹo & thủ thuật

- Kiểm tra kênh nhanh: `img.ndim` (2 = xám, 3 = màu) hoặc `len(img.shape)`.
- Histogram qua `cv2.calcHist([gray], [0], None, [256], [0, 256])` cho thấy phân bố độ sáng: lệch trái = thiếu sáng, lệch phải = thừa sáng, và hai đỉnh rõ rệt báo hiệu Otsu (`p1-binary`) sẽ chạy tốt.
- Tự tính bằng NumPy để kiểm chứng trọng số: `(0.114*img[:,:,0] + 0.587*img[:,:,1] + 0.299*img[:,:,2]).astype(np.uint8)`.
- Bẫy `uint8`: cộng trừ có thể bị tràn vòng (200 + 100 -> 44). Với phép tính trên pixel hãy ép sang `int16`/`float32`, hoặc dùng `cv2.add` (tự bão hòa ở 255).

## Thử ngay

- Upload ảnh có vùng đỏ tươi và vùng xanh lá cùng độ sáng (đèn giao thông, hoa): cả hai ra gần cùng một mức xám. Đây chính là lý do các bước sau dựa vào cường độ sáng thay vì màu.
- So với ảnh gốc, ảnh xám trông "cũ" hơn vì mất độ bão hòa: đó là phần thông tin đã bị nén đi.
