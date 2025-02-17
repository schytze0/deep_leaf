import os
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TRAIN_PATH, VALID_PATH, IMG_SIZE, BATCH_SIZE, KAGGLE_USERNAME, KAGGLE_KEY
import shutil

def setup_kaggle_auth():
    """
    Sets up Kaggle API authentication dynamically based on `.env` configuration.
    """
    # Directly use variables from config.py
    if KAGGLE_USERNAME and KAGGLE_KEY:
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = KAGGLE_KEY
        print("Kaggle API credentials successfully set.")
    else:
        print("Kaggle API credentials are MISSING. Ensure they are correctly set in the .env file.")

def download_dataset():
    """
    Downloads the dataset from Kaggle if it's not already available locally.
    """
    setup_kaggle_auth()  # Ensure Kaggle API authentication is set

    # updated to save data in the project
    dataset_path = os.path.expanduser("./data/raw")
    download_temp_path = os.path.expanduser("~/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset")

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("Downloading dataset from KaggleHub...")
        downloaded_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

        print(f"Dataset successfully downloaded to: {downloaded_path}")
        shutil.move(download_temp_path, dataset_path)
        print(f"Raw data successfully transfered to the project at: { dataset_path}")
    else:
        print("Dataset already exists at {dataset_path}. Skipping download.")


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

