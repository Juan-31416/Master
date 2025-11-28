import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Global variables to store images
input_images = []
panorama_result = None

def select_images():
    """
    Opens a file dialog to select between 2 and 3 images for panorama stitching
    """

    global input_images

    # Open file dialog to select multiple images
    file_paths = filedialog.askopenfilenames(
        title="Select 2 to 8 images",
        filetypes=[("Image files", "*.jpg +.jpeg *.png *.bmp")]
    )

    # Validate number of images
    if len(file_paths) > 8:
        messagebox.showerror("Error", "Please select a maximum of 8 images")
        return

    # Load Images
    input_images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is not None:
            input_images.append(img)
        else:
            messagebox.showerror("Error", f"Could not loaf image: {path}")
            return
    
    messagebox.showinfo("Success", f"{len(input_images)} images loaded successfully")
    btn_create_panorama.config(state=tk.NORMAL)
    btn_show_originals.config(state=tk.NORMAL)

def detect_and_describe(image, method='SIFT'):
    """
    Detect keypoints and computes descriptors using SIFT or ORB

    Args:
        imame: Intup image
        method: 'SIFT' or 'ORB' for feature detection

    Returns:
        keypoints: Detected keypoints
        descriptors: Computed descriptors
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Choose feature detector
    if method == 'SIFT':
        detector = cv2.SIFT_create()
    else:
        detector = CV2.ORB_create(nfeatures=2000)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    return keypoints, descriptors

def match_features(desc1, desc2, method='SIFT', ratio=0.75):
    """
    Matches features between two images using BFMatcher or FLANN

    Args:
        desc1: Descriptors from first image
        desc2: Descriptors from sencond image
        method: 'SIFT' or 'ORB' to choose matcher type
        ratio: Ratio test threshold for good matches

    Returns:
        good_matches: List of good matches
    """

    if method == 'SIFT':
        # FLANN matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # BFMatcher for ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Find k best matches
    matches = matcher.knnMatch(desc1, desc2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    
    return good_matches

def compute_homography(kp1, kp2, matches):
    """
    Computes homography matrix using RANSAC

    Args:
        kp1: Keypoints from first image
        kp2: Keypoints from second image
        matches: Good matches between images
    
    Returns:
        H: Homography matrix
        status: Inlier status for each match
    """

    # Need at least 4 matches to compute homography
    if len(matches) < 4:
        return None, None

    # Extract matched keypoint coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dts_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    H, status = cv2.findHomography(src_pts, dts_pts, cv2.RANSAC, 5.0)

    return H, status

def warp_images(img1, img2, H):
    """
    Warps img1 to align iwth img2 using homography matrix

    Args:
        img1: Image to be warped
        img2: Reference image
        H: Homography matrix

    Returns:
        result: Warped and blended panorama
        offset: Translation offset used
    """

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Get corners of img1
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    # Transform corners using homography
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Get corners of img2
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Combine all corners to find canvas size
    all_corners = np.concatenate((warped_corners, corners_img2), axis=0)

    # Find min and max coodinates
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Calculate translation to kepp all content visible
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # Apply translation to homography
    H_translated = translation.dot(H)

    # Calculate output size
    output_width = x_max - x_min
    output_height = y_max - y_min

    # Warp img1 to the new canvas
    warped_img1 = cv2.warpPerspective(img1, H_translated, (output_width, output_height))

    # Create canvas and place img2
    result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    result[-y_min:-y_min+h2, -x_min:-x_min+w2] = img2

    # Blend images in overlapping region
    # Create masks for blending
    mask1 = (warped_img1 > 0).astype(np.uint8)
    mask2 = (result > 0).astype(np.uint8)
    overlap_mask = cv2.bitwise_and(mask1, mask2)

    # Simple alpha blending in overlap region
    for c in range(3):
        overlap_region = overlap_mask[:, :, c] > 0
        if np.any(overlap_region):
            result[:, :, c][overlap_region] = (
                0.5 * warped_img1[:, :, c][overlap_region] + 0.5 * result[:, :, c][overlap_region]
            ).astype(np.uint8)
    
    # Add non-overlapping parts of warped_img1
    non_overlap = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))
    for c in range(3):
        result[:, :, c] = np.where(non_overlap[:, :, c] > 0, warped_img1[:, :, c], result[:, :, c])

    return result, (-x_min, -y_min)
    
def crop_black_borders(image):