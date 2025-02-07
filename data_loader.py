import os
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TRAIN_PATH, VALID_PATH, IMG_SIZE, BATCH_SIZE, KAGGLE_USERNAME, KAGGLE_KEY, ROOT_FOLDER

def setup_kaggle_auth():
    """
    Ensures Kaggle API authentication is set up dynamically using environment variables.
    """
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY

def download_dataset():
    """
    Downloads the dataset from Kaggle if it's not already available locally.
    """
    setup_kaggle_auth()  # Ensure Kaggle API authentication is set

    if not os.path.exists(ROOT_FOLDER):
        print("Downloading dataset from KaggleHub...")
        kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset", path=ROOT_FOLDER, force=False)
    else:
        print("Dataset already exists. Skipping download.")

def load_data():
    """
    Loads dataset using separate paths for training and validation.
    """
    download_dataset()  # Ensure the dataset is available

    datagen = ImageDataGenerator(rescale=1./255)

    train_data = datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    val_data = datagen.flow_from_directory(
        VALID_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_data, val_data

if __name__ == "__main__":
    # Test loading data
    train_data, val_data = load_data()
    print(f"Train samples: {train_data.samples}, Validation samples: {val_data.samples}")

