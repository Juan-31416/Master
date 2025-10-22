import cv2

# Open the default camera (0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")

else:
    while True:
        ret, frame = cap.read()  # Capture each frame
        if not ret:
            break

        cv2.imshow("Camera Frame", frame)  # Display each frame

        # Break on 'ESC' key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows