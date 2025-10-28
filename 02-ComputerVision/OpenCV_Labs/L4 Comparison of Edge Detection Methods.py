import cv2
import numpy as np

# Load the input image
img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error loading image")
    exit()

# Sobel Edge Detection
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined = cv2.convertScaleAbs(sobel_combined)

# Prewitt Edge Detection
prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewitt_x = cv2.filter2D(img, -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(img, -1, prewitt_kernel_y)
prewitt_combined = cv2.convertScaleAbs(prewitt_x + prewitt_y)

# Canny Edge Detection
canny_edges = cv2.Canny(img, 50, 150)

# Laplacian of Gaussian (LoG) Edge Detection (OpenCV Way)
sigma = 1
# kernel_size = (2 * int(3 * sigma) + 1, 2 * int(3 * sigma) + 1)  # Kernel size based on sigma
blurred = cv2.GaussianBlur(img, (0, 0), sigma, 0)  # Apply Gaussian Blur
log_edges = cv2.Laplacian(blurred, cv2.CV_64F)  # Apply Laplacian
log_edges = cv2.convertScaleAbs(log_edges)  # Convert to uint8 for visualization

# Difference of Gaussian (DoG) Edge Detection (OpenCV Way)
gaussian1 = cv2.GaussianBlur(img, (5, 5), 1)  # Gaussian with sigma=1
gaussian2 = cv2.GaussianBlur(img, (5, 5), 2)  # Gaussian with sigma=2
dog_edges = cv2.convertScaleAbs(gaussian1 - gaussian2)  # Difference of Gaussians

# Display results
cv2.imshow("Original Image", img)
cv2.imshow("Sobel Combined", sobel_combined)
cv2.imshow("Prewitt Combined", prewitt_combined)
cv2.imshow("Canny Edges", canny_edges)
cv2.imshow("Laplacian of Gaussian (LoG)", log_edges)
cv2.imshow("Difference of Gaussian (DoG)", dog_edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
