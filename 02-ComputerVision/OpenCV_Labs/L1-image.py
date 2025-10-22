#1. Import cv2 librery
import cv2
import os

#2. Print the version of the library
#print(cv2.__version__)

#3. Load an image
script_dir = os.path.dirname(__file__)
img_path = os.path.join(script_dir, 'img', 'dog.jpg')
img = cv2.imread(img_path)

if img is None:
    print(f'Error: Could not load image at: {img_path}')

else:
    #4. Resize the image to half its original size
    resized_img = cv2.resize(img, (800, 600))

    #5. Show the image
    cv2.imshow('Imagen', img)

    #6. Wait for a key to be pressed
    cv2.waitKey(0)

    #7. Close all windows
    cv2.destroyAllWindows()