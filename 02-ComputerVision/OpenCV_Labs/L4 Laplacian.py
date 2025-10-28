import cv2

# Load image
img = cv2.imread("colorful.jpg")#, cv2.IMREAD_GRAYSCALE)

# Laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)

# Display
cv2.imshow("Original", img)
cv2.imshow("Laplacian raw", laplacian)
cv2.imshow("Laplacian absolute", laplacian_abs)

cv2.waitKey(0)
cv2.destroyAllWindows()
