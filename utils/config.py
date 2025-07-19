import os

# Paths for the dataset and model output
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Hyperparameters for model training
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 10

# Paths to pre-trained models or checkpoints
PRETRAINED_MODEL_PATH = os.path.join(MODEL_DIR, "pretrained_model.pth")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")

# Logging and output settings
LOG_DIR = os.path.join(BASE_DIR, "logs")