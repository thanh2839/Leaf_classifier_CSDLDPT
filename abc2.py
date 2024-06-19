import cv2
import numpy as np
from matplotlib import pyplot as plt


# Đọc ảnh từ file
img_path = './RGB-leaf/1. Quercus suber/iPAD2_C01_EX06.JPG'
image = cv2.imread(img_path)

# Kiểm tra xem ảnh có được đọc thành công không
if image is None:
    raise ValueError("Image not found or unable to read")

# Chuyển ảnh sang thang độ xám
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Áp dụng Gaussian Blur để làm mịn ảnh và giảm nhiễu
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Sử dụng adaptive threshold để nhị phân hóa ảnh
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Áp dụng erosion để loại bỏ nhiễu
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(binary, kernel, iterations=1)

# Tìm các đường viền
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tạo một bản sao của ảnh gốc để vẽ đường viền
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Tạo một ảnh trắng có cùng kích thước với ảnh gốc
contour_binary = np.zeros_like(gray)

# Vẽ chỉ các đường viền lên ảnh đen trắng này
cv2.drawContours(contour_binary, contours, -1, 255, 1)

# Tạo một ảnh mới chỉ chứa phần viền của lá
leaf_contour = cv2.bitwise_and(eroded, contour_binary)

# Vẽ các đường viền lên ảnh trắng này
cv2.drawContours(contour_binary, contours, -1, 255, 1)

# Thiết lập giá trị pixel của phần viền của lá thành màu trắng (255)
leaf_contour_white = np.where(leaf_contour == 255, 255, 0)



# Hiển thị ảnh thang độ xám, ảnh đã làm mịn, ảnh nhị phân và ảnh với đường viền
plt.figure(figsize=(20, 5))
plt.subplot(1, 6, 1)
plt.title('Grayscale Image')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 6, 2)
plt.title('Blurred Image')
plt.imshow(blurred, cmap='gray')
plt.axis('off')

plt.subplot(1, 6, 3)
plt.title('Binary Image')
plt.imshow(eroded, cmap='gray')
plt.axis('off')

plt.subplot(1, 6, 4)
plt.title('Contour Image on Binary Image')
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Hiển thị ảnh contour trong ảnh đen trắng
# plt.figure(figsize=(10, 5))
# plt.title('Contour Image on Binary Background')
# plt.imshow(contour_binary, cmap='gray')
# plt.axis('off')

# Hiển thị ảnh contour trong ảnh đen trắng
plt.subplot(1, 6, 5)
plt.title('Contour Image on Binary Background')
plt.imshow(contour_binary, cmap='gray')
plt.axis('off')

plt.subplot(1, 6, 6)
plt.title('Leaf Contour Only')
plt.imshow(leaf_contour_white, cmap='gray')
plt.axis('off')
plt.show()

plt.show()
