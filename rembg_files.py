#Preprocessing Techniques

import torchvision.transforms as transforms

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 3. Crack Detection-Specific Enhancements
# a) Edge Detection
# Use edge-detection filters (like Sobel, Canny) to enhance crack boundaries.

import cv2
import numpy as np

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return edges



# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# Apply localized contrast enhancement for better crack visibility.

def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


# Noise Handling

def denoise_image(image):
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h=10, hForColor=10, templateWindowSize=7)
    return denoised


#  Fourier Transform

def apply_fourier_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum
