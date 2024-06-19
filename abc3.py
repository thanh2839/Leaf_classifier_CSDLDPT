import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = './RGB-leaf/22. Primula vulgaris/iPAD2_C22_EX01.JPG'
image = cv2.imread(image_path)

# Convert RGB to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply erosion to remove remaining noise
kernel = np.ones((5, 5), np.uint8)
eroded_image = cv2.erode(blurred_image, kernel, iterations=1)

# Convert the grayscale image to a binary image using Otsu's thresholding
ret, binary_image = cv2.threshold(eroded_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours
contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty black image to draw the contours
contour_image = np.zeros_like(binary_image)

# Draw the contours on the empty black image
cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

# Plot all the images
titles = ['RGB Image', 'Gray Scale Image', 'Blurred Image', 'Eroded Image', 'Binary Image', 'Contour Image']
images = [image, gray_image, blurred_image, eroded_image, binary_image, contour_image]

plt.figure(figsize=(12, 10))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray' if i != 0 else None)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
