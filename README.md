# AI nhập môn (2nd degree practices)

| Thư mục         | Nội dung                                                                          |
| --------------- | --------------------------------------------------------------------------------- |
| `docs/`         | Giáo trình Markdown, build bằng `mdsvr` và publish lên GitHub Pages               |
| `practices/`    | Code thực hành theo từng chương (`00-practice`, `01-opencv-co-ban`, ...)          |
| `cv-playground/`| Web app demo CV (FastAPI + React), build thành image lên GitHub Container Registry |

```bash
mdsvr docs                                  # xem docs local
docker run -p 8000:8000 ghcr.io/<owner>/<repo>/cv-playground:latest
```
