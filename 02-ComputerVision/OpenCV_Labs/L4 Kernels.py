import cv2
import numpy as np

# Load image
img = cv2.imread("lena.jpg")#, cv2.IMREAD_GRAYSCALE)

# Define custom kernels
sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
edge_enhance_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

# Apply filters
sharpened = cv2.filter2D(img, -1, sharpening_kernel)
edge_enhanced = cv2.filter2D(img, -1, edge_enhance_kernel)
embossed = cv2.filter2D(img, -1, emboss_kernel)

# Display results
cv2.imshow("Original", img)
cv2.imshow("Sharpened", sharpened)
cv2.imshow("Edge Enhanced", edge_enhanced)
cv2.imshow("Embossed", embossed)

cv2.waitKey(0)
cv2.destroyAllWindows()
