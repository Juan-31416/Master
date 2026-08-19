import cv2
import numpy as np

# Load two images
img1 = cv2.imread("Landscape1R.jpg")  # First image (left)
img2 = cv2.imread("Landscape2.jpg")  # Second image (right)

if img1 is None or img2 is None:
    print("Error loading images")
    exit()

if img1 is None or img2 is None:
    print("Error loading images")
    exit()

# Step 1: Detect features using SIFT
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Step 2: Match features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Step 3: Apply Lowe's ratio test
good_matches = [m for m, n in matches if m.distance < 0.3 * n.distance]

# Step 4: Extract matched points
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

# Step 5: Compute homography
H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

# Step 6: Warp the second image to align with the first
height, width = img1.shape[:2]
aligned_img = cv2.warpPerspective(img2, H, (width, height))

# Step 7: Create the panorama canvas
panorama = cv2.warpPerspective(img2, H, (width * 2, height))
panorama[0:height, 0:width] = img1  # Place the first image

# Step 8: Crop the black areas from the panorama
gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
_, thresh = cv2.threshold(gray_panorama, 1, 255, cv2.THRESH_BINARY)  # Threshold to create a binary mask
coords = cv2.findNonZero(thresh)  # Find all non-zero points
x, y, w, h = cv2.boundingRect(coords)  # Find the bounding box
cropped_panorama = panorama[y:y+h, x:x+w]  # Crop the image

# Step 9: Visualize matched points
matched_points_img = cv2.drawMatches(
    img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Step 10: Display the results
cv2.imshow("Input Images (Side by Side)", cv2.hconcat([img1, img2]))
cv2.imshow("Matched Points", matched_points_img)
cv2.imshow("Aligned Image", aligned_img)
cv2.imshow("Panorama", panorama)
cv2.imshow("Cropped Panorama", cropped_panorama)

# Save the cropped panorama
# cv2.imwrite("cropped_panorama.jpg", cropped_panorama)

cv2.waitKey(0)
cv2.destroyAllWindows()