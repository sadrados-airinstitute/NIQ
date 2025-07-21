from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException
from model.ocr_model import OCRModel
from model.entity_recognition_model import EntityRecognitionModel
from model.train import EntityRecognitionModelTrainer, ClassifierModelTrainer
from utils.logger import Logger
from io import BytesIO
from PIL import Image
import requests
import os
from typing import Optional
from model.nutrition_extraction_pipeline import NutritionExtractionPipeline
from utils.create_dataset import ClassifierDataset, EntityRecognitionDataset
from model.evaluation import ClassifierModelEvaluator, EntityRecognitionModelEvaluator
from model.classifier_model import ClassifierModel
from data_preprocessing.create_dataset import EntityRecognitionDataset
from model.entity_recognition_model import EntityRecognitionModel

class API:
    def __init__(self):
        self.logger = Logger(log_dir="logs", log_file="api.log")
        self.logger_instance = self.logger.get_logger()
        self.app = FastAPI()
        
    def create_endpoints(self):
        
        @self.app.get("/")
        def read_root():
            return {"message": "Hello, World!"}
        
        
        # Define the /start_training endpoint
        @self.app.post("/train_entity_recognition_model")
        async def start_training(background_tasks: BackgroundTasks, csv_path: str, model_path: Optional[str] = None, model_name: Optional[str] = None, epochs: Optional[int] = 10, learning_rate: Optional[float] = 5e-5):
            """
            Endpoint to start training the model in the background.

            Args:
                csv_path (str): Path to CSV file with training data.
                model_path (Optional[str]): Directory to save or load the model from.
                model_name (Optional[str]): Model identifier (e.g., "bert-base-cased").
                epochs (Optional[int]): Number of training epochs (default: 10).
                learning_rate (Optional[float]): Learning rate (default: 5e-5).

            Returns:
                dict: Confirmation message.
            """
            # Validate CSV file
            try:
                if not csv_path.endswith(".csv"):
                    raise ValueError("Provided training file is not a CSV.")
                if not os.path.isfile(csv_path):
                    raise FileNotFoundError(f"CSV file not found at: {csv_path}")
                
                # Validate epochs
                if epochs is None or epochs <= 0:
                    raise ValueError("Epochs must be a positive integer.")

                # Validate learning rate
                if learning_rate is None or learning_rate <= 0:
                    raise ValueError("Learning rate must be a positive float.")

                self.logger_instance.info(f"Initiating training with CSV: {csv_path}, epochs: {epochs}, learning_rate: {learning_rate}, model: {model_name}, save_to: {model_path}")

                er_model_trainer = EntityRecognitionModelTrainer(csv_path=csv_path, model_path=model_path, model_name=model_name, epochs=epochs, learning_rate=learning_rate)
                
                background_tasks.add_task(
                    er_model_trainer.train_model,
                )

                return {"message": f"Training has started in the background using CSV: {csv_path}, epochs: {epochs}, learning_rate: {learning_rate}, model: {model_name}, save_to: {model_path}"}

            except (ValueError, FileNotFoundError) as ve:
                self.logger_instance.error(f"Validation error: {str(ve)}")
                raise HTTPException(status_code=400, detail=str(ve))

            except Exception as e:
                self.logger_instance.error(f"Unexpected error: {str(e)}")
                raise HTTPException(status_code=500, detail="Error while starting training.")

        # Define the /start_training endpoint
        @self.app.post("/train_classifier_model")
        async def start_training_2(background_tasks: BackgroundTasks, csv_path: str, model_path: Optional[str] = None, model_name: Optional[str] = None, epochs: Optional[int] = 10, learning_rate: Optional[float] = 5e-5):
            """
            Endpoint to start training the model in the background.

            Args:
                csv_path (str): Path to CSV file with training data.
                model_path (Optional[str]): Directory to save or load the model from.
                model_name (Optional[str]): Model identifier (e.g., "bert-base-cased").
                epochs (Optional[int]): Number of training epochs (default: 10).
                learning_rate (Optional[float]): Learning rate (default: 5e-5).

            Returns:
                dict: Confirmation message.
            """
            # Validate CSV file
            try:
                if not csv_path.endswith(".csv"):
                    raise ValueError("Provided training file is not a CSV.")
                if not os.path.isfile(csv_path):
                    raise FileNotFoundError(f"CSV file not found at: {csv_path}")
                
                # Validate epochs
                if epochs is None or epochs <= 0:
                    raise ValueError("Epochs must be a positive integer.")

                # Validate learning rate
                if learning_rate is None or learning_rate <= 0:
                    raise ValueError("Learning rate must be a positive float.")

                self.logger_instance.info(f"Initiating training with CSV: {csv_path}, epochs: {epochs}, learning_rate: {learning_rate}, model: {model_name}, save_to: {model_path}")

                er_model_trainer = ClassifierModelTrainer(csv_path=csv_path, model_path=model_path, model_name=model_name, epochs=epochs, learning_rate=learning_rate)
                
                background_tasks.add_task(
                    er_model_trainer.train_model,
                )

                return {"message": f"Training has started in the background using CSV: {csv_path}, epochs: {epochs}, learning_rate: {learning_rate}, model: {model_name}, save_to: {model_path}"}

            except (ValueError, FileNotFoundError) as ve:
                self.logger_instance.error(f"Validation error: {str(ve)}")
                raise HTTPException(status_code=400, detail=str(ve))

            except Exception as e:
                self.logger_instance.error(f"Unexpected error: {str(e)}")
                raise HTTPException(status_code=500, detail="Error while starting training.")


        @self.app.post("/extract_nutritional_info_from_folder")
        def extract_nutritional_info_from_folder(image_folder_path: str, ner_checkpoint_dir: str, classifier_checkpoint_dir: str , languages: Optional[list] = ["en", "fr", "es", "de", "it"]):
            """
            Extracts nutritional information from all images in a local folder.

            Args:
                image_folder_path (str): Path to the folder containing image files.
                ner_checkpoint_dir (str): Path to the NER model checkpoint.
                classifier_checkpoint_dir (str): Path to the classifier model checkpoint.
                languages (list): List of languages for EasyOCR (ISO 639-1 codes).

            Returns:
                dict: Dictionary with extracted nutritional information per image.
            """ 
            try:
                if not os.path.isdir(folder_path):
                    raise HTTPException(status_code=400, detail=f"Folder does not exist: {folder_path}")

                supported_ext = (".jpg", ".jpeg", ".png")
                image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.lower().endswith(supported_ext)]

                if not image_paths:
                    raise HTTPException(status_code=404, detail="No image files found in the folder.")

                # Initialize the pipeline only once
                nutrition_info_extractor = NutritionExtractionPipeline(image_paths=image_paths, ner_checkpoint_dir=ner_checkpoint_dir, classifier_checkpoint_dir=classifier_checkpoint_dir, languages=languages)

                # Process all images
                results = nutrition_info_extractor.run()

                return {"results": results}

            except Exception as e:
                self.logger_instance.error(f"Error processing folder: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

        @self.app.post("/evaluate_classifier_model")
        def evaluate_classifier_model(csv_path: str, model_checkpoint_dir: str, max_length: int = 64, batch_size: int = 16):
            """
            Evaluates a trained binary classifier model (linked vs. not linked triplets).

            Args:
                csv_path (str): CSV with pre-labeled triplets and binary targets.
                model_checkpoint_dir (str): Directory with classifier model + tokenizer.
                max_length (int): Max token length.
                batch_size (int): Evaluation batch size.

            Returns:
                dict: Evaluation metrics (accuracy, precision, recall, F1).
            """
            try:
                
                # Load dataset
                dataset = ClassifierDataset(csv_path=csv_path, max_length=max_length)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                # Load model
                model_wrapper = ClassifierModel(checkpoint_dir=model_checkpoint_dir)
                evaluator = ClassifierModelEvaluator(model=model_wrapper.model, tokenizer=model_wrapper.tokenizer, dataloader=dataloader)

                metrics = evaluator.evaluate()

                return {"classifier_evaluation": metrics}

            except Exception as e:
                self.logger_instance.error(f"Error evaluating classifier: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
            
            
        @self.app.post("/evaluate_entity_recognition_model")
        def evaluate_ner_model(csv_path: str, model_checkpoint_dir: str, max_length: int = 128, batch_size: int = 8):
            """
            Evaluates a trained NER (BIO tagging) model on a validation set.

            Args:
                csv_path (str): CSV to reconstruct validation examples.
                model_checkpoint_dir (str): Directory with NER model, tokenizer, label2id.
                max_length (int): Max sequence length.
                batch_size (int): Evaluation batch size.

            Returns:
                dict: Token-level and entity-level metrics.
            """
            try:
                # Load NER model
                model_wrapper = EntityRecognitionModel(checkpoint_dir=model_checkpoint_dir)

                # Build dataset
                dataset = EntityRecognitionDataset(csv_path=csv_path, max_length=max_length)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                # Run evaluator
                evaluator = EntityRecognitionModelEvaluator(
                    model=model_wrapper.model,
                    label_map=model_wrapper.id2label,
                    tokenizer=model_wrapper.tokenizer,
                    evaluation_data_loader=dataloader
                )

                metrics = evaluator.evaluate(device=model_wrapper.device)

                return {"ner_evaluation": metrics}

            except Exception as e:
                self.logger_instance.error(f"Error evaluating NER model: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        # Testing logs and exception handling
        @self.app.get("/test_logs_and_exceptions")
        async def test_logs_and_exceptions():
            """
            A test endpoint to verify logging and exception handling.
            """
            try:
                # Logging at different levels
                self.logger_instance.info("Test INFO log: This is an info-level log.")
                self.logger_instance.warning("Test WARNING log: This is a warning-level log.")
                self.logger_instance.error("Test ERROR log: This is an error-level log.")
                
                # Raise an HTTPException to test error handling
                raise HTTPException(status_code=418, detail="I'm a teapot. Just testing HTTPException handling.")

            except HTTPException as he:
                # Log and raise the HTTPException
                self.logger_instance.error(f"HTTPException occurred: {str(he.detail)}")
                raise he
            except Exception as e:
                # Catch any other exceptions and log them
                self.logger_instance.error(f"Unexpected error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Unexpected error occurred: {str(e)}")