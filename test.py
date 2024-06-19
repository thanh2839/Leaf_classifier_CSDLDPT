import cv2
import numpy as np

# Read the image in RGB format
image = cv2.imread('./RGB-leaf/22. Primula vulgaris/iPAD2_C22_EX01.JPG')
mask = np.ones_like(image[:,:,0])  # Assuming a grayscale mask
# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Check mask data type
if mask.dtype != np.uint8:
    mask = mask.astype(np.uint8)

# Check mask size
if mask.shape != image.shape[:2]:
    mask = cv2.resize(mask, dsize=(image.shape[1], image.shape[0]))  # Resize to match image

# Define the lower and upper bounds of the green color range in HSV
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

# Create a mask for the green color range
mask = cv2.inRange(hsv_image, lower_green, upper_green)

# Invert the mask to get the background pixels
background_mask = np.invert(mask)

# Create a white background image with the same number of channels as the original image
white_background = np.ones_like(image) * 255

# Apply the background mask to each channel of the white background image
masked_background = white_background * background_mask[:, :, np.newaxis]

# Combine the masked background with the original image
final_image = cv2.bitwise_and(image, image, mask=masked_background)

# Display the final image
cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
