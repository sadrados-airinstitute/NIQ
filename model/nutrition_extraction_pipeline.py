from model.ocr_model import OCRModel
from model.entity_recognition_model import EntityRecognitionModel
from model.classifier_model import ClassifierModel
from typing import List, Tuple, Dict

class NutritionExtractionPipeline:
    def __init__(
        self,
        image_paths: str,
        ner_checkpoint_dir: str,
        classifier_checkpoint_dir: str,
        ocr_languages: List[str] = ['en', 'fr', 'es', 'de', 'it'],
        device: str = None
    ):
        """
        Initializes the pipeline and its internal models.

        Args:
            image_path (List[str]): String of the local path to nutrition images.
            ner_checkpoint_dir (str): Directory containing the trained NER model.
            classifier_checkpoint_dir (str): Directory containing the trained classifier.
            ocr_languages (List[str], optional): OCR languages to support. Defaults to multilingual.
            device (str, optional): 'cuda' or 'cpu'. Auto-detects if None.
        """
        self.image_paths = image_paths
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Instantiate models
        self.ocr_model = OCRModel(images_urls=image_paths, languages=ocr_languages)
        self.ner_model = EntityRecognitionModel(model_dir=ner_checkpoint_dir)
        self.classifier_model = ClassifierModel(checkpoint_dir=classifier_checkpoint_dir, device=self.device)

    def extract_candidate_triplets(self, tokens: List[str], labels: List[str]) -> List[Tuple[str, str, str]]:
        nutrients, quantities, units = [], [], []

        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label == "B-NUTRIENT":
                nutrients.append((i, token))
            elif label == "B-QUANTITY":
                quantities.append((i, token))
            elif label == "B-UNIT":
                units.append((i, token))

        triplets = []
        for n_idx, n_tok in nutrients:
            for q_idx, q_tok in quantities:
                for u_idx, u_tok in units:
                    if max([n_idx, q_idx, u_idx]) - min([n_idx, q_idx, u_idx]) <= 5:
                        triplets.append((n_tok, q_tok, u_tok))
        return triplets

    def normalize_nutrient(self, text: str) -> str:
        return text.strip().lower()

    def run(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Runs the full OCR → NER → Classifier pipeline for all images.

        Returns:
            Dict: A dictionary mapping each image to its extracted nutrition info.
        """
        output = {}
        ocr_results = self.ocr_model.process_all()

        for image_path, words in ocr_results.items():
            if not words:
                output[image_path] = {}
                continue

            ner_output = self.ner_model.predict(words)
            tokens, labels = zip(*ner_output) if ner_output else ([], [])

            triplets = self.extract_candidate_triplets(tokens, labels)

            result = {}
            for nutrient, quantity, unit in triplets:
                if self.classifier_model.predict(nutrient, quantity, unit) == 1:
                    nutrient_key = self.normalize_nutrient(nutrient)
                    try:
                        result[nutrient_key] = {
                            "value": float(quantity),
                            "unit": unit
                        }
                    except ValueError:
                        continue

            output[image_path] = result

        return output