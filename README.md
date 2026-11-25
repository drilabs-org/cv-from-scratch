# CV nhập môn

Lộ trình tự học Computer Vision qua project thực hành, dành cho người đã biết
Python nhưng chưa từng làm xử lý ảnh hay machine learning. Mỗi project tạo ra
một sản phẩm chạy được, độ khó tăng dần từ OpenCV thuần đến huấn luyện mạng
neuron.

- **Docs site**: https://drilabs-org.github.io/cv-from-scratch/
- **Docker image**: `ghcr.io/drilabs-org/cv-from-scratch/cv-playground`

## Cấu trúc

| Thư mục          | Nội dung                                                                        |
| ---------------- | ------------------------------------------------------------------------------- |
| `docs/`          | Giáo trình Markdown (3 chương), build bằng `mdsvr`, deploy lên GitHub Pages     |
| `practices/`     | Code thực hành theo từng chương (`01-opencv-co-ban`, `02-mnist-chu-so-viet-tay`, `03-bien-bao-giao-thong`) |
| `cv-playground/` | Web app demo các bài tập CV (FastAPI + React), đóng gói Docker                  |
| `_scripts/`      | Script hỗ trợ export docs (`fix-basepath.mjs`)                                  |
| `.github/`       | CI/CD: deploy docs lên Pages, build và push Docker image lên GHCR               |

## Chạy docs local

```bash
npx mdsvr docs            # xem docs tại localhost
npm run export            # export static site ra docs/_html/public
```

## Chạy cv-playground

Bằng Docker (khuyên dùng):

```bash
docker run -p 8000:8000 ghcr.io/drilabs-org/cv-from-scratch/cv-playground:latest
# mở http://localhost:8000
```

Hoặc chạy dev local (backend :8000 + frontend :5173):

```bash
cd cv-playground
./dev.sh
```

Muốn dùng weights MNIST tự train, có thể mount thư mục model vào app; hướng dẫn
nằm trong [CV Playground README](cv-playground/README.md#use-custom-mnist-weights).
Chi tiết kiến trúc và cách thêm task mới cũng xem tại đó.

## Release

Hai workflow trong `.github/workflows/`:

| Workflow     | Trigger                    | Kết quả                                       |
| ------------ | -------------------------- | --------------------------------------------- |
| `docs.yml`   | push `main` (thay đổi `docs/`) | Deploy docs site lên GitHub Pages             |
| `docker.yml` | push tag `v*`              | Build và push image lên GHCR (`latest`, semver, `sha-*`) |

Để release image mới:

```bash
git tag v0.2.0
git push origin v0.2.0
```

## License

MIT, xem [LICENSE](LICENSE).
