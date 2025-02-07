import os

# Kaggle API Credentials (loaded from environment variables)
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")

# Root directory for the dataset
ROOT_FOLDER = "/root/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2"
DATA_DIR = os.path.join(ROOT_FOLDER, "new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/")

# Define separate paths for training and validation
TRAIN_PATH = os.path.join(DATA_DIR, "train")
VALID_PATH = os.path.join(DATA_DIR, "valid")

# Training parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 20

