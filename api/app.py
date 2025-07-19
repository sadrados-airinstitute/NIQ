# api/app.py

from fastapi import FastAPI, BackgroundTasks, File, UploadFile
from model.nutritional_information_extraction_model import extract_nutritional_info_from_image
from model.train import train_entity_recognition_model
from io import BytesIO
from PIL import Image
import requests

class API:
    def __init__(self):
        self.app = FastAPI()  # Initialize the FastAPI app
    
    def create_endpoints(self):
        # Define the /extract_nutritional_info_from_image endpoint
        @self.app.post("/extract_nutritional_info_from_image")
        async def extract_nutritional_info_from_image_endpoint(image: UploadFile = File(...)):
            """
            Endpoint to extract nutritional information from an uploaded image.
            """
            # Read image from the upload
            image_content = await image.read()
            image_pil = Image.open(BytesIO(image_content))

            # Extract nutritional info from the image
            nutritional_info = extract_nutritional_info_from_image(image_pil)

            return {"nutritional_info": nutritional_info}

        # Define the /start_training endpoint
        @self.app.post("/start_training")
        async def start_training(background_tasks: BackgroundTasks, epochs: int = 10):
            """
            Endpoint to start training the model in the background.
            """
            background_tasks.add_task(train_entity_recognition_model, epochs=epochs)
            return {"message": f"Training has started in the background for {epochs} epochs."}

        # Define the /extract_nutritional_info_from_url endpoint
        @self.app.post("/extract_nutritional_info_from_url")
        async def extract_info_from_url(url: str):
            """
            Endpoint to extract nutritional information from an image URL.
            """
            response = requests.get(url)
            image_pil = Image.open(BytesIO(response.content))
            nutritional_info = extract_nutritional_info_from_image(image_pil)
            return {"nutritional_info": nutritional_info}
