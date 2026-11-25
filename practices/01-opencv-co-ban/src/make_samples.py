import cv2
import numpy as np

img = np.full((400, 400, 3), 255, dtype=np.uint8)
cv2.rectangle(img, (30, 30), (150, 150), (200, 60, 40), -1)
cv2.circle(img, (300, 100), 60, (60, 180, 75), -1)
cv2.putText(img, "CV", (120, 320), cv2.FONT_HERSHEY_SIMPLEX,
            3, (30, 30, 30), 8)
cv2.imwrite("sample.png", img)
print("Da luu: sample.png")

# Anh dong xu gia lap de test lenh `count` (thay bang anh chup that sau)
coins = np.full((400, 600), 40, dtype=np.uint8)
positions = [(90, 90), (300, 80), (510, 95), (100, 300),
             (300, 310), (500, 290), (200, 190), (420, 190)]
for i, (x, y) in enumerate(positions):
    cv2.circle(coins, (x, y), 35 + i % 3 * 5, 200, -1)
cv2.imwrite("coins.jpg", coins)
print("Da luu: coins.jpg")
