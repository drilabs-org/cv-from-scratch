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


def count_objects(img, min_area=500, ksize=5, method="otsu", invert="auto",
                  dilate=0, block=11, C=2):
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
