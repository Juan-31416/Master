import cv2
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "./img/googleCar.jpg")

# Load the image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error loading image")
    exit()

# Apply Canny edge detector
low_threshold = 50
high_threshold = 2 * low_threshold
kernel_size = 3
canny_edges = cv2.Canny(img, low_threshold, high_threshold, apertureSize=kernel_size)#,L2gradient=True)

# Display results
cv2.imshow("Original", img)
cv2.imshow("Canny Edges", canny_edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
