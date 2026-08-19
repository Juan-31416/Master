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
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
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
            messagebox.showerror("Error", f"Could not load image: {path}")
            return
    
    messagebox.showinfo("Success", f"{len(input_images)} images loaded successfully")
    btn_create_panorama.config(state=tk.NORMAL)
    btn_show_originals.config(state=tk.NORMAL)

def detect_and_describe(image, method='SIFT'):
    """
    Detect keypoints and computes descriptors using SIFT or ORB

    Args:
        image: Intup image
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
        detector = cv2.ORB_create(nfeatures=2000)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    return keypoints, descriptors

def match_features(desc1, desc2, method='SIFT', ratio=0.75):
    """
    Matches features between two images using BFMatcher or FLANN

    Args:
        desc1: Descriptors from first image
        desc2: Descriptors from second image
        method: 'SIFT' or 'ORB' to choose matcher type
        ratio: Ratio test threshold for good matches

    Returns:
        good_matches: List of good matches
    """

    if method == 'SIFT':
        # FLANN matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # BFMatcher for ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Find k best matches
    try:
        matches = matcher.knnMatch(desc1, desc2, k=2)
    except Exception as e:
        print(f"Error in knnMatch: {e}")
        return []

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
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H, status

def warp_images(img1, img2, H):
    """
    Warps img1 to align with img2 using homography matrix

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

    # Find min and max coordinates
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
    """
    Crops black borders from the panorama to minimize black areas
    
    Args:
        image: Input panorama image
    
    Returns:
        cropped: Cropped image without black borders
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to find non-black pixels
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Get bounding rectangle of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop image
        cropped = image[y:y+h, x:x+w]
        return cropped
    
    return image

def create_panorama():
    """
    Main function to create panorama images form inputs
    """

    global panorama_result

    if len(input_images) < 2:
        messagebox.showerror("Error", "Please load images first (2 min and 8 max)")
        return

    try:
        # Use SIFT for feature detection
        method = 'SIFT'

        # Start with the middle image as base
        base_idx = len(input_images) // 2
        result = input_images[base_idx].copy()

        # Stitch images to the right
        for i in range(base_idx + 1, len(input_images)):
            print(f"Stitching image {i+1}/{len(input_images)}...")

            # Detect features
            kp1, desc1 = detect_and_describe(result, method)
            kp2, desc2 = detect_and_describe(input_images[i], method)

            # Match features
            matches = match_features(desc1, desc2, method)

            if len(matches) < 4:
                messagebox.showwarning("Warning", f"Not enough matches for image {i+1}")
                continue

            # Compute homography
            H, status = compute_homography(kp1, kp2, matches)

            if H is None:
                messagebox.showwarning("Warning", f"Could not compute homography for image {i+1}")
                continue

            # Warp and blend
            result, offset = warp_images(result, input_images[i], H)
        
        # Stitch images to the left
        for i in range(base_idx - 1, -1, -1):
            print(f"Stitching image {i+1}/{len(input_images)}...")

            # Detect features
            kp1, desc1 = detect_and_describe(input_images[i], method)
            kp2, desc2 = detect_and_describe(result, method)

            # Match features
            matches = match_features(desc1, desc2, method)

            if len(matches) < 4:
                messagebox.showwarning("Warning", f"Not enough matches for image {i+1}")
                continue

            # Compute homography
            H, status = compute_homography(kp1, kp2, matches)

            if H is None:
                messagebox.showwarning("Warning", f"Could not compute homography for image {i+1}")
                continue

            # Warp and blend
            result, offset = warp_images(input_images[i], result, H)
        
        # Crop black borders
        result = crop_black_borders(result)

        panorama_result = result

        messagebox.showinfo("Success", f"Panorama created successfully!")
        btn_show_result.config(state=tk.NORMAL)
    
    except Exception as e:
        messagebox.showerror("Error", f"Error creating panorama: {str(e)}")
    
def show_original_images():
    """
    Displays all original input images in a new window
    """
    if len(input_images) == 0:
        messagebox.showerror("Error", "No images loaded")
        return
    
    # Create new window
    window = tk.Toplevel(root)
    window.title("Original Images")

    # Calculate grid layout
    cols = min(3, len(input_images))
    rows = (len(input_images) + cols -1) // cols

    # Display each image
    for idx, img in enumerate(input_images):
        row = idx // cols
        col = idx % cols

        # Resize for display
        display_img = cv2.resize(img, (300, 200))
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        # Convert to PhotoImage
        img_pil = Image.fromarray(display_img)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Create label
        label = tk.Label(window, image=img_tk)
        label.image = img_tk # Keep reference
        label.grid(row=row+1, column=col)

        # Add image number
        text_label = tk.Label(window, text=f"Image {idx+1}")
        text_label.grid(row=row, column=col)

def show_panorama_result():
    """
    Displays the final panorama result in a new window
    """

    if panorama_result is None:
        messagebox.showerror("Error", "No panorama created yet")
        return
    
    # Create new window
    window = tk.Toplevel(root)
    window.title("Panorama Result")

    # Resize for display if too large
    h, w = panorama_result.shape[:2]
    max_width = 1200
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        display_img = cv2.resize(panorama_result, (new_w, new_h))
    else:
        display_img = panorama_result.copy()
    
    # Convert to RGB
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

    # Convert to PhotoImage
    img_pil = Image.fromarray(display_img)
    img_tk = ImageTk.PhotoImage(img_pil)
    
    # Create label
    label = tk.Label(window, image=img_tk)
    label.image = img_tk  # Keep reference
    label.pack(padx=10, pady=10)

    # Add save button
    def save_panorama():
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if file_path:
            cv2.imwrite(file_path, panorama_result)
            messagebox.showinfo("Success", "Panorama saved successfully!")
        
    btn_save = tk.Button(window, text="Save Panorama", command=save_panorama, bg="#4CAF50", fg="white", font=("Arial", 12))
    btn_save.pack(pady=10)

# Create main window
root = tk.Tk()
root.title("Panorama Image Stitching - Computer Vision")
root.geometry("500x300")
root.configure(bg="#f0f0f0")

# Title label
title_label = tk.Label(root, text="Panoramic Image Stitching", 
                        font=("Arial", 18, "bold"),
                        bg="#f0f0f0")
title_label.pack(pady=10)

#Instructuions label
instructions = tk.Label(root, text="Select 2 to 8 images to create a panorama", font=("Arial", 11), bg="#f0f0f0")
instructions.pack(pady=5)

# Button frame
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=20)

# Select-images button
btn_select = tk.Button(button_frame, text="Select Images",
                                command=select_images, 
                                bg="#2196F3", fg="white", font=("Arial", 12), 
                                width=15, height=2)
btn_select.grid(row=0, column=0, padx=10, pady=5)

# Show-original-images button
btn_show_originals = tk.Button(button_frame, text="Show Originals", 
                               command=show_original_images,
                               bg="#FF9800", fg="white", font=("Arial", 12),
                               width=15, height=2, state=tk.DISABLED)
btn_show_originals.grid(row=0, column=1, padx=10, pady=5)

# Create panorama button
btn_create_panorama = tk.Button(button_frame, text="Create Panorama", 
                               command=create_panorama,
                               bg="#4CAF50", fg="white", font=("Arial", 12),
                               width=15, height=2, state=tk.DISABLED)
btn_create_panorama.grid(row=1, column=0, padx=10, pady=5)

# Show result button
btn_show_result = tk.Button(button_frame, text="Show Result", 
                           command=show_panorama_result,
                           bg="#9C27B0", fg="white", font=("Arial", 12),
                           width=15, height=2, state=tk.DISABLED)
btn_show_result.grid(row=1, column=1, padx=10, pady=5)
# Footer label
footer_label = tk.Label(root, text="Computer Vision - Master CCAM", 
                       font=("Arial", 9), bg="#f0f0f0", fg="gray")
footer_label.pack(side=tk.BOTTOM, pady=10)

# Run the application
root.mainloop()