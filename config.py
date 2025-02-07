import os
import datetime

# Kaggle API Credentials (User-Specific)
GITHUB_ACTOR = os.getenv("GITHUB_ACTOR", "default_user").upper()  # Detect GitHub username

KAGGLE_USERNAME = os.getenv(f"KAGGLE_USERNAME_{GITHUB_ACTOR}", "")
KAGGLE_KEY = os.getenv(f"KAGGLE_KEY_{GITHUB_ACTOR}", "")

# Root directory for the dataset
ROOT_FOLDER = "/root/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2"
DATA_DIR = os.path.join(ROOT_FOLDER, "new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/")

# Define paths for training, validation, and testing
TRAIN_PATH = os.path.join(DATA_DIR, "train")
VALID_PATH = os.path.join(DATA_DIR, "valid")
TEST_PATH = os.path.join(ROOT_FOLDER, "test/test"

# Model save path
MODEL_DIR = "models"  # Directory to store trained models
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")  # Path for saving/loading the model

# Training parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 50

# Ensure the model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Logging & History paths
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure logs directory exists

# Generate timestamp for unique history files
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
HISTORY_PATH = os.path.join(LOGS_DIR, f"history_{TIMESTAMP}.json")
