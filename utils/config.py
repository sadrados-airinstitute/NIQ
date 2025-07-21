import os

# Data Loader
MAX_CONCURRENT_REQUESTS = 200
MAX_RETRIES = 3
RETRY_DELAY = 3
TIMEOUT = 10
IMAGE_SAVE_FOLDER = 'data/downloaded_images/'

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


NUTRITION_DICTIONARY = {
    'energy': ['energy', 'energia', 'energie', 'energiegehalt', 'kj', 'kilojoules', 'kjoule'],
    'kcal-energy': ['calories', 'calorie', 'kcal', 'kcalories', 'calorias', 'caloria', 'kalorien', 'kalorie', 'kcals', 'kcalorie', 'kcaloria'],
    'proteins': ['protein', 'proteins', 'proteine', 'proteines', 'proteina', 'proteinas'],
    'fat': ['fat', 'fats', 'gras', 'grasas', 'lipides', 'grassi'],
    'saturated-fat': ['saturated', 'saturatedfat', 'saturates','saturados', 'grasassaturadas', 'satures', 'graissessaturees','saturi', 'grassisaturi', 'gesaettigte', 'gesaettigtesfett', 'gesaettigtefettsaeuren'],
    'monounsaturated-fat': ['monounsaturated', 'monounsaturatedfat', 'monoinsaturadas', 'grasasmonoinsaturadas', 'monoinsaturees', 'graissesmonoinsaturees', 'monoinsaturi', 'grassimonoinsaturi', 'einfachungesaettigte', 'einfachfettsaeuren', 'einfachfett'],
    'polyunsaturated-fat': ['polyunsaturated', 'polyunsaturatedfat', 'poliinsaturadas', 'grasaspoliinsaturadas', 'polyinsaturees', 'graissespolyinsaturees', 'grassipolinsaturi', 'polinsaturi', 'mehrfachungesaettigte', 'mehrfachfettsaeuren','mehrfachfett'],
    'trans-fat': ['trans', 'transfat', 'grasastrans', 'graissestrans', 'grassitrans', 'transfette'],
    'carbohydrates': ['carbs', 'carbohydrates', 'glucides', 'hidratos', 'carboidrati'],
    'sugars': ['sugar', 'sugars', 'sucres', 'azucares', 'zuccheri'],
    'added-sugars': ['added', 'addedsugars', 'azucaresanadidos', 'sucresajoutes', 'zuccheriaggiunti', 'zugesetzterzucker'],
    'fiber': ['fiber', 'fibers', 'fibra', 'fibres', 'fibre'],
    'salt': ['salt', 'sel', 'sal', 'sale']
}

LABEL_LIST = {
    'O', 'B-UNIT', 'B-QUANTITY', 'B-ENERGY', 'B-KCAL-ENERGY', 'B-PROTEINS', 'B-FAT', 'B-SATURATED-FAT', 'B-MONOUNSATURATED-FAT','B-POLYUNSATURATED-FAT', 'B-TRANS-FAT', 'B-CARBOHYDRATES', 'B-SUGARS', 'B-ADDED-SUGARS','B-FIBER','B-SALT'
}


VALID_UNITS = {
    # Mass
    'g', 'mg', 'kg', 'mcg', 'µg', 'ug'
    # Volume
    'ml', 'l', 'cl', 'dl',
    # Energy
    'kcal', 'cal', 'kj',
    # Count-based or other
    'unit', 'unidades', 'porcion', 'portion', 'serving', 'servings', '%'
}