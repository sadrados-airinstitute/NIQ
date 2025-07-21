import re
import unicodedata
from Levenshtein import distance
from fuzzywuzzy.fuzz import ratio
from typing import Optional, Dict, List, Tuple
from utils.config import NUTRITION_DICTIONARY, VALID_UNITS


def normalize_and_split(token: str) -> list:
    """
    Normalize and tokenize a string by removing accents, converting to lowercase,
    replacing commas with dots, and splitting into alphanumeric segments.

    Args:
        token (str): Raw token string (e.g., OCR output).

    Returns:
        List[str]: List of normalized subcomponents (e.g., ['proteins', '11.3']).
    """
    token = unicodedata.normalize('NFKD', token)
    token = token.encode('ascii', 'ignore').decode('utf-8').lower()
    token = token.replace(',', '.')
    parts = re.findall(r'[a-zA-Z]+|\d+\.\d+|\d+', token)
    return parts


def label_token_for_ner(token: str, threshold_fuzzy: int = 80) -> str:
    """
    Assign a BIO label to a token based on fuzzy and levenshtein match to nutrition keywords.

    Args:
        token (str): The token to classify.
        threshold_fuzzy (int): Minimum fuzzy ratio to accept a match.

    Returns:
        str: The BIO NER label, or 'O' if no match is above the threshold.
    """
    token = token.lower()
    best_score = 0
    best_key = None

    for nutrient_key, variants in NUTRITION_DICTIONARY.items():
        for variant in variants:
            score = ratio(token, variant)
        
            if score > best_score and score >= threshold_fuzzy:
                best_score = score
                best_key = nutrient_key

    if best_key:
        return f"B-{best_key.upper()}"
    else:
        return "O"

def ocr_output_dict_to_ner_dict(ocr_dict: Dict[str, List[str]], threshold_fuzzy: int = 80) -> Dict[str, Dict[str, List[str]]]:
    """
    Converts a dictionary of OCR results into a dictionary with NER annotations in BIO format.

    Args:
        ocr_dict (Dict[str, List[str]]): Dictionary where keys are image paths and values are OCR token lists.
        threshold_fuzzy (int): Similarity threshold for fuzzy matching nutrient names.

    Returns:
        Dict[str, Dict[str, List[str]]]: Dictionary with image paths as keys and values like:
            {
                "tokens": [...],
                "labels": [...]
            }
    """
    ner_output = {}

def ocr_output_dict_to_ner_dict(ocr_dict: Dict[str, List[str]], threshold_fuzzy: int = 80) -> Dict[int, Dict[str, List[str]]]:
    """
    Converts OCR results into a dictionary with NER BIO-formatted annotations,
    using the numeric image index as key.

    Args:
        ocr_dict (Dict[str, List[str]]): Dictionary where keys are image paths and values are OCR token lists.
        threshold_fuzzy (int): Similarity threshold for fuzzy nutrient name matching.

    Returns:
        Dict[int, Dict[str, List[str]]]: Dictionary indexed by image number, each entry contains:
            {
                "tokens": [...],
                "labels": [...]
            }
    """
    ner_output = {}

    for image_path, ocr_tokens in ocr_dict.items():
        # Extract numeric index from the filename (e.g., "../image_123.jpg" -> 123)
        base_name = os.path.basename(image_path)
        match = re.search(r'image_(\d+)\.jpg', base_name)
        if not match:
            continue  # skip files that don't match the expected pattern
        image_index = int(match.group(1))

        tokens = []
        labels = []

        for raw_token in ocr_tokens:
            sub_tokens = normalize_and_split(raw_token)

            for sub in sub_tokens:
                if re.fullmatch(r'\d+(\.\d+)?', sub):  # Numerical quantity
                    tokens.append(sub)
                    labels.append("B-QUANTITY")

                elif sub.lower() in VALID_UNITS:  # Valid unit
                    tokens.append(sub)
                    labels.append("B-UNIT")

                else:  # Potential nutrient name
                    nutrient_label = label_token_for_ner(sub, threshold_fuzzy=threshold_fuzzy)
                    tokens.append(sub)
                    labels.append(nutrient_label)

        ner_output[image_index] = {
            "tokens": tokens,
            "labels": labels
        }

    return ner_output
