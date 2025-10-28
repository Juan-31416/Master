import cv2

# Load the image
img = cv2.imread("moon_shot_noise.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error loading image")
    exit()

# Apply filters
blurred = cv2.blur(img, ksize=(5, 5))  # Simple averaging
gaussian = cv2.GaussianBlur(img, (3, 3), 0)  # Gaussian blur
median = cv2.medianBlur(img, 5)  # Median blur
box_filtered = cv2.boxFilter(img, ddepth=-1, ksize=(5, 5))

# Display results
cv2.imshow("Original", img)
cv2.imshow("Averaging (Blur)", blurred)
cv2.imshow("Gaussian Blur", gaussian)
cv2.imshow("Median Blur", median)
cv2.imshow("Box Filter", box_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()