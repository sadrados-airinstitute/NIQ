from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline
from utils.utils import load_from_checkpoint_entity_recognition
from typing import List, Tuple

class EntityRecognitionModel:
    """
    EntityRecognitionModel loads a trained NER model from checkpoint and performs inference.
    """

    def __init__(self, model_dir: str = 'models/ner'):
        """
        Initializes the model using a loading function.

        Args:
            model_dir (str): Path to the directory containing model, tokenizer, and label maps.
            device (str): Device to use (e.g., 'cuda', 'cpu'). If None, auto-selects.
        """

        self.model, self.tokenizer, self.label2id, self.id2label = load_from_checkpoint_entity_recognition(model_dir)
        
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()
        pass

    def predict(self, text: str) -> List[Tuple[str, str]]:
        """
        Performs NER inference on the input text.

        Args:
            text (str): Text from OCR.

        Returns:
            List of (token, predicted_label) tuples.
        """
        if isinstance(text, str):
            encoding = self.tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=False)
            tokens = self.tokenizer.tokenize(text)
            word_ids = encoding.word_ids()
        else:
            encoding = self.tokenizer(text, is_split_into_words=True, return_tensors="pt", truncation=True)
            tokens = text
            word_ids = encoding.word_ids()

        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

        final_preds = []
        seen = set()
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen:
                continue
            label_id = predictions[idx]
            label = self.id2label[label_id]
            final_preds.append((tokens[word_idx], label))
            seen.add(word_idx)

        return final_preds
