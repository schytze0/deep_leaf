import os

# Kaggle API Credentials (loaded from environment variables)
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")

# Root directory for the dataset
ROOT_FOLDER = "/root/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2"
DATA_DIR = os.path.join(ROOT_FOLDER, "new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/")

# Define paths for training and validation
TRAIN_PATH = os.path.join(DATA_DIR, "train")
VALID_PATH = os.path.join(DATA_DIR, "valid")

# Model save path
MODEL_DIR = "models"  # Directory to store trained models
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.h5")  # Path for saving/loading the model

# Training parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 20

# Ensure the model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
