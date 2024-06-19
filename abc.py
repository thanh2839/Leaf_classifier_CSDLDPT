import cv2
import numpy as np
from matplotlib import pyplot as plt

# Đọc ảnh từ file
img_path = './RGB-leaf/1. Quercus suber/iPAD2_C01_EX01.JPG'
image = cv2.imread(img_path)

# Kiểm tra xem ảnh có được đọc thành công không
if image is None:
    raise ValueError("Image not found or unable to read")

# Chuyển ảnh sang thang độ xám
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Áp dụng erosion để loại bỏ nhiễu
kernel = np.ones((5, 5), np.uint8)
eroded = cv2.erode(gray, kernel, iterations=1)

# Chuyển ảnh sang nhị phân (binary image) sử dụng ngưỡng nhị phân hóa
_, binary = cv2.threshold(eroded, 127, 255, cv2.THRESH_BINARY)

# Tìm các đường viền
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tạo một bản sao của ảnh gốc để vẽ đường viền
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Hiển thị ảnh thang độ xám, ảnh đã loại bỏ nhiễu, ảnh nhị phân và ảnh với đường viền
plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.title('Grayscale Image')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Eroded Image')
plt.imshow(eroded, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Binary Image')
plt.imshow(binary, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Contour Image')
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
