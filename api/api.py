from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException
from model.nutritional_information_extraction_model import NutritionInfoExtractor
from model.ocr_model import OCRModel
from model.entity_recognition_model import EntityRecognitionModel
from model.train import EntityRecognitionModelTrainer
from utils.logger import Logger
from io import BytesIO
from PIL import Image
import requests

class API:
    def __init__(self):
        self.logger = Logger(log_dir="logs", log_file="api.log")
        self.logger_instance = self.logger.get_logger()
        self.ocr_model = OCRModel()
        self.entity_recognition_model = EntityRecognitionModel()
        self.nutrition_info_extractor = NutritionInfoExtractor(ocr_model=self.ocr_model, ner_model=self.entity_recognition_model)
        self.entity_recognition_model_trainer = EntityRecognitionModelTrainer()
        self.app = FastAPI()
        
    def create_endpoints(self):
        
        @self.app.get("/")
        def read_root():
            return {"message": "Hello, World!"}
        
        
        # Define the /start_training endpoint
        @self.app.post("/start_training")
        async def start_training(background_tasks: BackgroundTasks, epochs: int = 10):
            """
            Endpoint to start training the model in the background.

            Args:
                epochs (int): The number of epochs to train the model for.

            Returns:
                dict: Confirmation message.
            """
            try:
                # Ensure epochs are positive
                if epochs <= 0:
                    raise ValueError("Epochs must be a positive integer.")
                
                self.logger_instance.info(f"Starting model training with {epochs} epochs.")
                background_tasks.add_task(self.model_trainer.train_model, epochs=epochs)
                
                self.logger_instance.info(f"Model training task added to the background with {epochs} epochs.")
                return {"message": f"Training has started in the background for {epochs} epochs."}

            except ValueError as ve:
                self.logger_instance.error(f"Invalid epochs value: {str(ve)}")
                raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
            except Exception as e:
                self.logger_instance.error(f"Error starting training: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")

        # Define the /extract_nutritional_info_from_image endpoint
        @self.app.post("/extract_nutritional_info_from_image")
        async def extract_nutritional_info_from_image_endpoint(image: UploadFile = File(...)):
            """
            Endpoint to extract nutritional information from an uploaded image.

            Args:
                image (UploadFile): The image file containing nutritional information.

            Returns:
                dict: The extracted nutritional information.
            """
            try:
                # Validate if file is an image
                if not image.content_type.startswith('image/'):
                    self.logger_instance.error("Uploaded file is not an image.")
                    raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

                # Read image content
                image_content = await image.read()
                image_pil = Image.open(BytesIO(image_content))

                # Log the image processing
                self.logger_instance.info("Extracting nutritional information from uploaded image.")

                # Extract nutritional info
                nutritional_info = self.nutrition_info_extractor.extract_nutritional_info_from_image(image_pil)

                # Log successful extraction
                self.logger_instance.info("Successfully extracted nutritional information from image.")

                return {"nutritional_info": nutritional_info}

            except HTTPException as he:
                # Log the error if it's an HTTPException
                self.logger_instance.error(f"HTTP error: {str(he.detail)}")
                raise he
            except Exception as e:
                # General error logging for image processing
                self.logger_instance.error(f"Error extracting nutritional information: {str(e)}")
                raise HTTPException(status_code=500, detail="Error processing the image.")

        @self.app.post("/extract_nutritional_info_from_url")
        async def extract_info_from_url(url: str):
            """
            Endpoint to extract nutritional information from an image URL.
            """
            try:
                # Log that the URL was received
                self.logger_instance.info(f"Received request to extract info from URL: {url}")

                response = requests.get(url, timeout=10)
                response.raise_for_status()  # Check if the URL fetch was successful

                # Check if content is an image
                if "image" not in response.headers["Content-Type"]:
                    self.logger_instance.warning(f"The URL content is not an image: {url}")
                    raise HTTPException(status_code=400, detail="The URL does not point to an image.")

                # Process the image
                image_pil = Image.open(BytesIO(response.content))

                # Extract nutritional info
                nutritional_info = self.nutrition_info_extractor.extract_nutritional_info_from_image(image_pil)

                # Log successful extraction
                self.logger_instance.info(f"Successfully processed nutritional info from URL: {url}")
                return {"nutritional_info": nutritional_info}

            except requests.exceptions.Timeout:
                self.logger_instance.error(f"Request to the URL timed out: {url}")
                raise HTTPException(status_code=408, detail="Request to the URL timed out.")
            except requests.exceptions.RequestException as e:
                self.logger_instance.error(f"Error fetching the image from URL: {url}, Error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error fetching the image from URL: {str(e)}")
            except Exception as e:
                self.logger_instance.error(f"Unexpected error while processing the URL: {url}, Error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


        # Define the /test_logs_and_exceptions endpoint (testing logs and exception handling)
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