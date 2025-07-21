import cv2
import numpy as np
import easyocr
from typing import List, Dict
from data_preprocessing.image_preprocessing import preprocess_image, rotate_image, download_images_from_dataframe
from utils.config import IMAGE_SAVE_FOLDER
import pandas as pd
from typing import Optional, List
import os

class OCRModel:
    def __init__(self,images_path: str = IMAGE_SAVE_FOLDER, df: Optional[pd.DataFrame] = None, url_column: str = "image_nutrition_url", languages: List[str] = ['en', 'fr', 'es', 'de', 'it']):
        """
        Initializes the OCR model. You can either provide a DataFrame to download images,
        or set a local folder of already downloaded images.

        Args:
            images_path (str): Path to local image directory. Defaults to 'data/downloaded_images'.
            df (pd.DataFrame, optional): DataFrame containing image URLs to download and process.
            url_column (str): Column name in the DataFrame that contains image URLs.
            languages (List[str]): List of languages for EasyOCR. Default supports multilingual European labels.
        """
        self.languages = languages
        self.reader = easyocr.Reader(languages, verbose=False)

        # Download images if a DataFrame is provided
        if df is not None:
            if url_column not in df.columns:
                raise ValueError(f"Column '{url_column}' not found in DataFrame.")
            download_images_from_dataframe(df[url_column].tolist(), save_folder=images_path)

        # Ensure the folder exists
        if not os.path.isdir(images_path):
            raise FileNotFoundError(f"Image directory '{images_path}' not found.")
        
        self.images_folder = images_path
        
    def get_best_orientation(self, image):
        """
        Determine the optimal orientation of the input image based on OCR confidence.

        The function rotates the image through 0°, 90°, 180°, and 270°, performs OCR on each,
        and selects the orientation with the highest average confidence score.

        Args:
            image (np.ndarray): Preprocessed grayscale image.

        Returns:
            Tuple[int, List[Tuple], np.ndarray]:
                - best_angle (int): Angle (0, 90, 180, 270) with the highest OCR confidence.
                - best_result (list): OCR results at the best angle (bounding box, text, confidence).
                - rotated_final (np.ndarray): Image rotated to the best angle.
        """
        
        rotations = [0, 90, 180, 270]
        best_conf = 0
        best_angle = 0
        best_result = []

        for angle in rotations:
            rotated = rotate_image(image, angle)
            results = self.reader.readtext(rotated)
            if results:
                avg_conf = np.mean([conf for _, _, conf in results])
                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_angle = angle
                    best_result = results

        # Return the best angle, corresponding OCR results, and the rotated image
        rotated_final = rotate_image(image, best_angle)
        return best_angle, best_result, rotated_final

    def process_all(self) -> Dict[str, Dict]:
        """
        Return a dictioanry with each image's path and the word's list from OCR.

        Returns:
            Dict[str, List[str]]
        """
        
        output = {}

        for path in self.images_folder:
            img = cv2.imread(path)
            if img is None:
                print(f"Image not found or cannot be read: {path}")
                continue

            preprocessed = preprocess_image(img)
            _, results, _ = self.get_best_orientation(preprocessed)

            words = [text for _, text, _ in results]
            output[path] = words

        return output
