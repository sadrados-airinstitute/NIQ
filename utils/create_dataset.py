from typing import Optional, List, Dict, Tuple
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from fastapi import HTTPException

from utils.logger import Logger
from model.ocr_model import OCRModel
from data_preprocessing.text_preprocessing import ocr_output_dict_to_ner_dict
from model.entity_recognition_model import EntityRecognitionModel


class EntityRecognitionDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        examples: Optional[List[Dict[str, List[str]]]] = None,
        label2id: Optional[Dict[str, int]] = None,
        max_length: int = 128
    ):
        """
        Dataset class for training a NER model to extract nutritional information
        from OCR-processed food label images or directly from provided examples.

        Args:
            csv_path (str, optional): Path to CSV file containing image URLs for OCR processing.
            examples (List[Dict], optional): Precomputed list of token/label dictionaries.
            label2id (Dict, optional): Label-to-ID mapping. Required if `examples` is provided.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.max_length = max_length

        # Option 1: Load from provided examples
        if examples is not None:
            if label2id is None:
                raise ValueError("If 'examples' is provided, you must also provide 'label2id'.")
            self.examples = examples
            self.label2id = label2id

        # Option 2: Load from CSV and generate via OCR + NER preprocessing
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
            from model.ocr_model import OCRModel  # local import to avoid circular dependency
            from data_preprocessing.text_preprocessing import ocr_output_dict_to_ner_dict

            # OCR
            self.ocr_model = OCRModel(images_urls=self.df['image_nutrition_url'].tolist())
            try:
                ocr_dict = self.ocr_model.process_all()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

            # NER preprocessing
            self.ner_dict = ocr_output_dict_to_ner_dict(ocr_dict=ocr_dict)
            self.examples = list(self.ner_dict.values())

            # Build label map
            unique_labels = sorted({label for ex in self.examples for label in ex["labels"]})
            self.label2id = {label: i for i, label in enumerate(unique_labels)}

        else:
            raise ValueError("You must provide either 'csv_path' or 'examples'.")

    def __len__(self):
        """
        Returns:
            int: Number of examples available in the dataset.
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Prepare a single data sample for NER model input.

        Args:
            idx (int): Index of the data sample.

        Returns:
            dict: A dictionary with tokenized input IDs, attention mask, and aligned NER labels.
        """
        example = self.examples[idx]
        tokens = example["tokens"]
        labels = example["labels"]

        # Convert label strings to integer IDs
        label_ids = [self.label2id[label] for label in labels]

        # Tokenize input using pre-tokenized words (split into words)
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Align token-level labels with wordpiece-level tokens
        word_ids = tokenized.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:
                aligned_labels.append(label_ids[word_idx])  # Label first sub-token
            else:
                aligned_labels.append(-100)  # Ignore subsequent sub-tokens
            previous_word_idx = word_idx

        # Return dictionary in Hugging Face-compatible format
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }
        
        
class ClassifierDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str] = None,
        examples: Optional[List[Tuple[str, str, str, int]]] = None,
        max_length: int = 128,
        ner_model_path: Optional[str] = "models/ner/",
        tokenizer_name: str = "bert-base-cased"
    ):
        """
        Dataset for training a binary classifier to detect valid (nutrient, quantity, unit) links.

        Args:
            csv_path (str): CSV path containing image URLs and ground truth structured nutrition info.
            examples (List[Tuple]): Optional list of (nutrient, quantity, unit, label) pairs.
            max_length (int): Max token sequence length.
            ner_model_path (str): Path to NER model for inference.
            tokenizer_name (str): Tokenizer to use for BERT-based classification.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        if examples is not None:
            self.data = examples

        elif csv_path is not None:
            # Cargar CSV y procesar imágenes con OCR
            self.df = pd.read_csv(csv_path)

            ocr_model = OCRModel(images_urls=self.df['image_nutrition_url'].tolist())
            try:
                ocr_dict = ocr_model.process_all()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

            ner_model = EntityRecognitionModel(model_dir=ner_model_path)

            all_examples = []
            for path, ocr_tokens in ocr_dict.items():
                ner_output = ner_model.predict(ocr_tokens)
                tokens, labels = zip(*ner_output) if ner_output else ([], [])

                triplets = self.extract_candidate_triplets(tokens, labels)
                truth = self.get_ground_truth_triplets(path)

                for triplet in triplets:
                    label = 1 if triplet[:3] in truth else 0
                    all_examples.append((*triplet[:3], label))

            self.data = all_examples
        else:
            raise ValueError("You must provide either 'csv_path' or 'examples'.")

    def extract_candidate_triplets(self, tokens: List[str], labels: List[str]) -> List[Tuple[str, str, str, str]]:
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
                        span = " ".join(tokens[min([n_idx, q_idx, u_idx]): max([n_idx, q_idx, u_idx]) + 1])
                        triplets.append((n_tok, q_tok, u_tok, span))

        return triplets

    def get_ground_truth_triplets(self, image_path: str) -> List[Tuple[str, str, str]]:
        """
        Extracts ground truth triplets for a given image from the dataframe.

        Returns:
            List[Tuple[str, str, str]]: known (nutrient, quantity, unit) triplets.
        """
        row = self.df[self.df["image_nutrition_url"] == image_path]
        if row.empty:
            return []

        triplets = []
        nutrient_cols = [col for col in row.columns if "_value" in col]
        for col in nutrient_cols:
            base = col.replace("_value", "")
            val = row[col].values[0]
            unit = row.get(f"{base}_unit", None).values[0] if f"{base}_unit" in row else None
            if pd.notna(val) and pd.notna(unit):
                triplets.append((base.lower(), str(val), unit))
        return triplets

    def __len__(self):
        """
        Returns:
            int: Number of examples available in the dataset.
        """
        return len(self.examples)

def __getitem__(self, idx):
    """
    Prepare a single data sample for a binary classification model input.

    Args:
        idx (int): Index of the data sample.

    Returns:
        dict: A dictionary with input IDs, attention mask, and binary label.
    """
    example = self.examples[idx]

    nutrient = example["nutrient"]
    quantity = example["quantity"]
    unit = example["unit"]
    label = int(example["label"])  # 0 or 1

    # Second segment: quantity + unit (e.g., "10 g")
    text_b = f"{quantity} {unit}"

    # Tokenize pair (nutrient, quantity+unit)
    encoding = self.tokenizer(
        nutrient,
        text_b,
        truncation=True,
        padding="max_length",
        max_length=self.max_length,
        return_tensors="pt"
    )

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": torch.tensor(label, dtype=torch.long)
    }