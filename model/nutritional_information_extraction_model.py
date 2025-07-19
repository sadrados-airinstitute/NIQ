from .ocr_model import extract_text_from_image
from .entity_recognition_model import extract_nutritional_entities


class NutritionalInfoExtractor:
    def __init__(self, ocr_model, ner_model):
        self.ocr_model = ocr_model  # OCR model instance
        self.ner_model = ner_model  # NER model instance

    def extract_nutritional_info_from_image(self, image_path: str):
        """
        Extracts nutritional information (e.g., calories, fat, sugars) from an image.
        
        Args:
            image_path (str): Path to the image file containing nutritional label.
        
        Returns:
            dict: A dictionary containing the nutritional information.
        """
        # Step 1: Extract text using OCR
        text = self.ocr_model.extract_text_from_image(image_path)
        
        # Step 2: Extract nutritional entities from the text
        entities = self.ner_model.extract_entities(text)
        
        # Step 3: Organize the extracted entities into a structured format (e.g., dictionary)
        nutritional_info = {}
        for entity in entities:
            # Example: We can classify and store the entities into a dictionary
            if "calories" in entity['word'].lower():
                nutritional_info['calories'] = entity['word']
            if "fat" in entity['word'].lower():
                nutritional_info['fat'] = entity['word']
            # Add other nutritional labels here (e.g., sugars, proteins)
        
        return nutritional_info

    def combine_results(self, text, entities):
        # Combine text and entity recognition results
        pass
