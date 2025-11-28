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

