import cv2

# Load the video
video = cv2.VideoCapture('C-HR_Car_Sensors.avi')

# Check if the video was loaded successfully
if not video.isOpened():
    print("Error: Could not open video.")

else:
    while True:
        ret, frame = video.read()  # Read each frame
        if not ret:
            break

        # Resize the frame to half its original size
        resized_frame = cv2.resize(frame, (800, 600))

        # cv2.imshow("Video Frame", frame)  # Display each frame
        cv2.imshow("Video Frame", resized_frame)  # Display each frame

        # Break on 'ESC' key
        if cv2.waitKey(30) == ord('q'):
        #if cv2.waitKey(30) & 0xFF == 27:
            break

    video.release()  # Release the video
    cv2.destroyAllWindows()  # Close all windows 