import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os

# Configurable parameters
#CAMARA = 0  # 0 for webcam, o route to video: 'video.mp4'
CAMARA = 'IMG_4329.MP4'
ALTURA_DISPLAY = 150  # height of each frame in pixels
KERNEL_GAUSS = (5, 5)  # Gaussian kernel dimension
SIGMA_GAUSS = 1.0  # Gaussian filter sigma
SOBEL_KSIZE = 3  # Dimension of kernel sobel
CANNY_TH1 = 50  # threshold 1 canny
CANNY_TH2 = 150  # threshold 2 canny
MOSTRAR_HISTOGRAMAS = True  # True o False

# Verify if the video file exist (only if camera not used)
if isinstance(CAMARA, str):
    if not os.path.exists(CAMARA):
        print(f"ERROR: El archivo '{CAMARA}' no existe")
        print(f"Directorio actual: {os.getcwd()}")
        print(f"Archivos en el directorio: {os.listdir('.')}")
        exit()
    else:
        print(f"Archivo encontrado: {CAMARA}")

# Open the camera or video file
cap = cv2.VideoCapture(CAMARA)

# Check if it is openning correctly
if not cap.isOpened():
    print("No se puede abrir la camara")
    exit()

print("Presiona 'q' para salir")

while True:
    # Read frame
    ret, frame = cap.read()
    
    if not ret:
        print("No se puede recibir frame")
        break
    
    ###################################
    # 1. Convert to grey scale & HSV  #
    ###################################

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    ###########################
    # 2. Calculate histograms #
    ###########################

    if MOSTRAR_HISTOGRAMAS:
        # RGB Histogram
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
        
        # HGrey scale histogram
        hist_gris = cv2.calcHist([gris], [0], None, [256], [0, 256])
        
        # HSV histogram
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    
    ############################
    # 3. Apply Gaussian filter #
    ############################

    frame_gauss = cv2.GaussianBlur(frame, KERNEL_GAUSS, SIGMA_GAUSS)
    gris_gauss = cv2.GaussianBlur(gris, KERNEL_GAUSS, SIGMA_GAUSS)
    hsv_gauss = cv2.GaussianBlur(hsv, KERNEL_GAUSS, SIGMA_GAUSS)
    
    #########################
    #  4. Border detection  #
    #########################
    
    # SOBEL in RGB
    gris_rgb = cv2.cvtColor(frame_gauss, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gris_rgb, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
    sobely = cv2.Sobel(gris_rgb, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
    sobel_rgb = np.sqrt(sobelx**2 + sobely**2)
    sobel_rgb = np.uint8(sobel_rgb * 255 / np.max(sobel_rgb))
    
    # SOBEL in grey scale
    sobelx_g = cv2.Sobel(gris_gauss, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
    sobely_g = cv2.Sobel(gris_gauss, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
    sobel_gris = np.sqrt(sobelx_g**2 + sobely_g**2)
    sobel_gris = np.uint8(sobel_gris * 255 / np.max(sobel_gris))
    
    # SOBEL in HSV
    hsv_v = hsv_gauss[:,:,2]
    sobelx_h = cv2.Sobel(hsv_v, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
    sobely_h = cv2.Sobel(hsv_v, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
    sobel_hsv = np.sqrt(sobelx_h**2 + sobely_h**2)
    sobel_hsv = np.uint8(sobel_hsv * 255 / np.max(sobel_hsv))
    
    # CANNY in RGB
    canny_rgb = cv2.Canny(gris_rgb, CANNY_TH1, CANNY_TH2)
    
    # CANNY in grey scale
    canny_gris = cv2.Canny(gris_gauss, CANNY_TH1, CANNY_TH2)
    
    # CANNY in HSV
    canny_hsv = cv2.Canny(hsv_v, CANNY_TH1, CANNY_TH2)
    
    # LAPLACIAN OF GAUSSIAN (LoG) in RGB
    log_rgb = cv2.Laplacian(gris_rgb, cv2.CV_64F)
    log_rgb = np.uint8(np.absolute(log_rgb))
    
    # LoG in grey scale
    log_gris = cv2.Laplacian(gris_gauss, cv2.CV_64F)
    log_gris = np.uint8(np.absolute(log_gris))
    
    # LoG in HSV
    log_hsv = cv2.Laplacian(hsv_v, cv2.CV_64F)
    log_hsv = np.uint8(np.absolute(log_hsv))
    
    ################
    # 5. Show all  #
    ################

    # Fit into screen
    h = ALTURA_DISPLAY
    w = int(h * frame.shape[1] / frame.shape[0])
    
    # Originals
    frame_small = cv2.resize(frame, (w, h))
    gris_small = cv2.resize(cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR), (w, h))
    hsv_small = cv2.resize(hsv, (w, h))
    
    # Filtered
    frame_gauss_small = cv2.resize(frame_gauss, (w, h))
    gris_gauss_small = cv2.resize(cv2.cvtColor(gris_gauss, cv2.COLOR_GRAY2BGR), (w, h))
    hsv_gauss_small = cv2.resize(hsv_gauss, (w, h))
    
    # Sobel
    sobel_rgb_small = cv2.resize(cv2.cvtColor(sobel_rgb, cv2.COLOR_GRAY2BGR), (w, h))
    sobel_gris_small = cv2.resize(cv2.cvtColor(sobel_gris, cv2.COLOR_GRAY2BGR), (w, h))
    sobel_hsv_small = cv2.resize(cv2.cvtColor(sobel_hsv, cv2.COLOR_GRAY2BGR), (w, h))
    
    # Canny
    canny_rgb_small = cv2.resize(cv2.cvtColor(canny_rgb, cv2.COLOR_GRAY2BGR), (w, h))
    canny_gris_small = cv2.resize(cv2.cvtColor(canny_gris, cv2.COLOR_GRAY2BGR), (w, h))
    canny_hsv_small = cv2.resize(cv2.cvtColor(canny_hsv, cv2.COLOR_GRAY2BGR), (w, h))
    
    # LoG
    log_rgb_small = cv2.resize(cv2.cvtColor(log_rgb, cv2.COLOR_GRAY2BGR), (w, h))
    log_gris_small = cv2.resize(cv2.cvtColor(log_gris, cv2.COLOR_GRAY2BGR), (w, h))
    log_hsv_small = cv2.resize(cv2.cvtColor(log_hsv, cv2.COLOR_GRAY2BGR), (w, h))
    
    # Assemble horizontally
    fila1 = np.hstack((frame_small, gris_small, hsv_small))
    fila2 = np.hstack((frame_gauss_small, gris_gauss_small, hsv_gauss_small))
    fila3 = np.hstack((sobel_rgb_small, sobel_gris_small, sobel_hsv_small))
    fila4 = np.hstack((canny_rgb_small, canny_gris_small, canny_hsv_small))
    fila5 = np.hstack((log_rgb_small, log_gris_small, log_hsv_small))
    
    # Assemble vertically
    resultado = np.vstack((fila1, fila2, fila3, fila4, fila5))
    
    # Put text
    cv2.putText(resultado, 'RGB', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(resultado, 'Grayscale', (w+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(resultado, 'HSV', (2*w+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(resultado, 'Original', (10, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(resultado, 'Gaussian', (10, 2*h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(resultado, 'Sobel', (10, 3*h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(resultado, 'Canny', (10, 4*h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(resultado, 'LoG', (10, 5*h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imshow('Procesamiento de Video', resultado)
    
    # Show histograms if activated
    if MOSTRAR_HISTOGRAMAS:
        # Create histogram plots
        fig1, ax1 = plt.subplots(1, 3, figsize=(10, 2))
        ax1[0].plot(hist_b, color='b')
        ax1[0].set_title('Blue')
        ax1[1].plot(hist_g, color='g')
        ax1[1].set_title('Green')
        ax1[2].plot(hist_r, color='r')
        ax1[2].set_title('Red')
        plt.tight_layout()
        canvas1 = FigureCanvasAgg(fig1)
        canvas1.draw()
        hist_rgb_img = np.frombuffer(canvas1.buffer_rgba(), dtype=np.uint8)
        hist_rgb_img = hist_rgb_img.reshape(fig1.canvas.get_width_height()[::-1] + (4,))
        hist_rgb_img = cv2.cvtColor(hist_rgb_img, cv2.COLOR_RGBA2BGR)
        cv2.imshow('Histograma RGB', hist_rgb_img)
        plt.close(fig1)
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 2))
        ax2.plot(hist_gris, color='gray')
        ax2.set_title('Grayscale')
        plt.tight_layout()
        canvas2 = FigureCanvasAgg(fig2)
        canvas2.draw()
        hist_gris_img = np.frombuffer(canvas2.buffer_rgba(), dtype=np.uint8)
        hist_gris_img = hist_gris_img.reshape(fig2.canvas.get_width_height()[::-1] + (4,))
        hist_gris_img = cv2.cvtColor(hist_gris_img, cv2.COLOR_RGBA2BGR)
        cv2.imshow('Histograma Gris', hist_gris_img)
        plt.close(fig2)
        
        fig3, ax3 = plt.subplots(1, 3, figsize=(10, 2))
        ax3[0].plot(hist_h)
        ax3[0].set_title('Hue')
        ax3[1].plot(hist_s)
        ax3[1].set_title('Saturation')
        ax3[2].plot(hist_v)
        ax3[2].set_title('Value')
        plt.tight_layout()
        canvas3 = FigureCanvasAgg(fig3)
        canvas3.draw()
        hist_hsv_img = np.frombuffer(canvas3.buffer_rgba(), dtype=np.uint8)
        hist_hsv_img = hist_hsv_img.reshape(fig3.canvas.get_width_height()[::-1] + (4,))
        hist_hsv_img = cv2.cvtColor(hist_hsv_img, cv2.COLOR_RGBA2BGR)
        cv2.imshow('Histograma HSV', hist_hsv_img)
        plt.close(fig3)
    
    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberate resources
cap.release()
cv2.destroyAllWindows()