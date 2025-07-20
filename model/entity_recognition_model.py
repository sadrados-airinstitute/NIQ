from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline

class EntityRecognitionModel:

    def __init__(self, model=None):
        """
        Initializes the EntityRecognitionModel. If no model is provided,
        it defaults to a pre-trained BERT model fine-tuned for NER (dbmdz/bert-large-cased-finetuned-conll03-english).
        
        Args:
            model: The pre-trained or custom model for NER. If None, the default BERT model is loaded.
        """
        if model is None:
            # Load a pre-trained BERT model for token classification (NER)
            self.model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
            self.tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
        else:
            # Use the provided custom model
            self.model = model
            self.tokenizer = BertTokenizer.from_pretrained(model)  # Assuming tokenizer is also passed with the model
        
        # Initialize the NER pipeline
        self.nlp_pipeline = pipeline('ner', model=self.model, tokenizer=self.tokenizer)

    def extract_nutritional_entities(self, text: str, labels=None):
        """
        Extracts nutritional entities from the OCR-extracted text using a pre-trained NER model.

        Args:
            text (str): The OCR-extracted text.
            labels (list): List of desired labels to filter entities (optional). Default is None.

        Returns:
            list: A list of recognized entities with labels and values.
        """
        # Perform NER to extract entities
        entities = self.nlp_pipeline(text)
        
        # If labels are provided, filter the entities based on the specified labels
        if labels:
            nutritional_entities = [entity for entity in entities if entity['label'] in labels]
        else:
            # If no labels provided, return all entities
            nutritional_entities = entities
        
        return nutritional_entities
    
    def preprocess_text(self, text: str):
        """
        Preprocesses the input text for better NER performance.
        E.g., you can implement tokenization, stopword removal, etc.

        Args:
            text (str): The raw OCR-extracted text.

        Returns:
            str: The cleaned or tokenized text.
        """
        # Clean and preprocess the text (this is just a placeholder)
        # You can implement your custom text preprocessing logic here
        # Example: Remove unwanted characters, lowercasing, etc.
        cleaned_text = text.lower()  # Example: convert text to lowercase
        return cleaned_text
