import pytesseract
import re
from PIL import Image
from .image_processor import ImageProcessor

class OCRProcessor:
    def __init__(self):
        self.image_processor = ImageProcessor()
        # Set tesseract path (Windows)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR"""
        try:
            # Preprocess image
            processed_img = self.image_processor.preprocess_image(image_path)
            
            # Configure OCR
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),-. '
            
            # Extract text
            text = pytesseract.image_to_string(processed_img, config=config)
            
            return self.clean_extracted_text(text)
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def clean_extracted_text(self, text):
        """Clean and normalize extracted text"""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except commas, parentheses, hyphens
        text = re.sub(r'[^\w\s,()-.]', '', text)
        
        # Convert to lowercase for consistency
        text = text.lower().strip()
        
        return text
    
    def extract_ingredients_list(self, text):
        """Extract ingredients from the full text"""
        # Look for ingredients section
        ingredients_patterns = [
            r'ingredients?[:\-\s]*(.*?)(?=directions|instructions|warning|caution|usage|$)',
            r'inci[:\-\s]*(.*?)(?=directions|instructions|warning|caution|usage|$)',
            r'composition[:\-\s]*(.*?)(?=directions|instructions|warning|caution|usage|$)'
        ]
        
        for pattern in ingredients_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                ingredients_text = match.group(1)
                # Split by commas and clean
                ingredients = [ing.strip() for ing in ingredients_text.split(',')]
                return [ing for ing in ingredients if len(ing) > 2]
        
        # If no specific section found, assume entire text is ingredients
        return [ing.strip() for ing in text.split(',') if len(ing.strip()) > 2]