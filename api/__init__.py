from fastapi import FastAPI, BackgroundTasks
from model.train import train_model  # Import the train function
from data_preprocessing.data_loader import get_data_loaders  # Import data loader function
from model import nutritional_information_extraction_model

app = FastAPI()

# Define the endpoint to trigger training
@app.post("/start_training")
async def start_training(background_tasks: BackgroundTasks):
    # Step 1: Initialize the data loaders
    train_loader, val_loader = get_data_loaders()

    # Step 2: Define the model
    model = nutritional_information_extraction_model.MultimodalModel()

    # Step 3: Add the training task to the background
    background_tasks.add_task(train_model, model, train_loader, val_loader, num_epochs=10)

    return {"message": "Training has started in the background!"}


