import cv2
import numpy as np
from PIL import Image

class ImageProcessor:
    def __init__(self):
        pass
    
    def preprocess_image(self, image_path):
        """Preprocess image for better OCR accuracy"""
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        
        # Resize if image is too small
        height, width = contrast.shape
        if width < 1000:
            scale_factor = 1000 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            contrast = cv2.resize(contrast, (new_width, new_height))
        
        return contrast
    
    def extract_text_regions(self, image):
        """Extract regions likely to contain text"""
        # Apply morphology to detect text regions
        kernel = np.ones((1, 5), np.uint8)
        img_dilation = cv2.dilate(image, kernel, iterations=1)
        img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
        
        return img_erosion