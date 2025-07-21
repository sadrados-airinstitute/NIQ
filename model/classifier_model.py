import torch
from typing import Union
from utils.utils import load_from_checkpoint_classifier  # adjust path if needed


class ClassifierModel:
    """
    ClassifierModel loads a trained binary classifier to determine whether
    a (nutrient, quantity, unit) triplet represents a valid relation (linked).
    """

    def __init__(self, checkpoint_dir: str, device: str = None):
        """
        Initializes the classifier model using a checkpoint directory.

        Args:
            checkpoint_dir (str): Path to the directory where the classifier model is saved.
            device (str): Device to use ('cuda' or 'cpu'). Auto-selects if None.
        """
        self.model, self.tokenizer = load_from_checkpoint_classifier(checkpoint_dir)

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, nutrient: str, quantity: str, unit: str) -> int:
        """
        Predicts whether the (nutrient, quantity, unit) triplet is linked.

        Args:
            nutrient (str): Nutrient name (e.g., "Protein")
            quantity (str): Quantity (e.g., "10")
            unit (str): Unit (e.g., "g")

        Returns:
            int: 1 if linked, 0 if not linked
        """
        text_b = f"{quantity} {unit}"
        inputs = self.tokenizer(
            nutrient,
            text_b,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        return prediction