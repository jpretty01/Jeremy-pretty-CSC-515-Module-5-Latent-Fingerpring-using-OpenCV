# Jeremy Pretty CSC 515 Module 5 Latent Fingerprint
# March 15, 2023
import cv2
import os


fingerprint = os.path.join(os.path.dirname(__file__), 'fingerprint.jpeg')

# Load the input image
image = cv2.imread(fingerprint)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding to obtain a binary image
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Define kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Apply dilation
dilated = cv2.dilate(binary, kernel, iterations=1)

# Apply erosion
eroded = cv2.erode(dilated, kernel, iterations=1)

# Apply opening
opening = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)

# Apply closing
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# Display the original fingerprint image, Dilated, and eroded image
cv2.imshow("Original", image)
cv2.imshow("Dilated", dilated)
cv2.imshow("Eroded", eroded)
cv2.waitKey(0)

# Display the enhanced fingerprint image
cv2.imshow("Enhanced Fingerprint", closing)
cv2.waitKey(0)

# Save the enhanced fingerprint image
cv2.imwrite("enhanced_fingerprint.jpg", closing)
