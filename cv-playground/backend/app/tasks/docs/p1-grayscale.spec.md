# Spec: Convert to grayscale (`p1-grayscale`)

Demo reference. This is the simplest task: there are no tunable controls, so this page documents the I/O contract and how to use the task in demos.

## I/O contract

- **Input**: any image `cv2.imdecode` can read (JPG, PNG, BMP, WebP...), decoded as a BGR `uint8`, 3-channel array.
- **Output**: single-channel `uint8` grayscale PNG (`output="image"`), no `data`.
- **Result tab**: `auto` -> lands on Output. The Compare slider shows the color/gray difference directly.

## Parameters

None. The params form is empty and Run sends `params={}`.

## Demo notes

- The conversion is `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`, i.e. $Y = 0.299R + 0.587G + 0.114B$, not a plain average: green contributes ~59%, red ~30%, blue ~11%.
- Best demo images: colorful scenes where channel weighting is visible. A saturated red object can come out surprisingly dark because red weighs only ~30%.
- Pipeline role: every other p1 task calls `to_gray` internally as its first stage, so this demo is "stage 0" of all the others.
- If your uploaded file fails to decode (`400 file cannot be decoded`), convert it to JPG/PNG first: `cv2.imdecode` does not read HEIC, TIFF variants it does not support, or RAW formats.
