import cv2
import numpy as np
from matplotlib import pyplot as plt

# Đọc ảnh RGB
img_rgb = cv2.imread('./RGB-leaf/1. Quercus suber/iPAD2_C01_EX04.JPG')

# Chuyển đổi ảnh sang thang độ xám
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Loại bỏ nhiễu bằng cách sử dụng mòn
kernel = np.ones((5, 5), np.uint8)
img_eroded = cv2.erode(img_gray, kernel, iterations=2)

# Phân đoạn ảnh nhị phân
_, img_bw = cv2.threshold(img_eroded, 100, 255, cv2.THRESH_BINARY)

# Tìm đường viền
contours, _ = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    approx = cv2.approxPolyDP(contour, approx_epsilon=0.1 * cv2.arcLength(contour, True), closed=True)
    cv2.drawContours(img_rgb, [approx], -1, (0, 255, 0), 3)

# Vẽ đường viền trên ảnh gốc
cv2.drawContours(img_rgb, contours, -1, (0, 255, 0), 3)

# Hiển thị kết quả
# cv2.imshow('Ảnh gốc', img_rgb)
# cv2.imshow('Ảnh thang độ xám', img_gray)
# cv2.imshow('Ảnh sau khi loại bỏ nhiễu', img_eroded)
# cv2.imshow('Ảnh nhị phân', img_bw)
# cv2.imshow('Đường viền', img_rgb)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.title('anh goc')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('do xam')
plt.imshow(img_gray)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('loại bỏ nhiễu')
plt.imshow(img_eroded)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('nhị phân')
plt.imshow(img_bw)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('đường viền')
plt.imshow(img_rgb)
plt.axis('off')

plt.show()
