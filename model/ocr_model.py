import pytesseract
from PIL import Image

class OCRModel:
    
    def __init__(self):
        # Initialize the OCR model (e.g., Tesseract or a custom deep learning model)
        pass
    
    def extract_text_from_image(image_path: str) -> str:
        """
        Uses Tesseract OCR to extract text from an image.

        Args:
            image_path (str): Path to the image file.
        
        Returns:
            str: Extracted text from the image.
        """
        # Open image
        image = Image.open(image_path)
        
        # Use Tesseract OCR to extract text
        text = pytesseract.image_to_string(image)
        
        return text

    def preprocess_image(self, image):
        # Preprocess image before feeding it to the OCR model (e.g., resizing, converting to grayscale)
        pass