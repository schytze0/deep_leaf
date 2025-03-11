import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO

# user defined functions and libraries
# from src.data_loader import load_data # TODO: needs to be written
from src.config import MODEL_PATH, IMG_SIZE, TEST_PATH
from src.helpers import load_tfrecord_data

def load_trained_model():
    '''
    Loads the trained model from disk.
    
    Returns:
    - model: The loaded Keras model.
    '''
    # TODO: NEEDS UPDATE TO NEW MODEL models/production_model.keras
    return tf.keras.models.load_model(MODEL_PATH)

def get_class_labels():
    '''
    Retrieves class labels dynamically from the training data.

    Returns:
    - class_labels (dict): Mapping from index to class label.
    '''

    # new data load from .tfrecord
    # DEBUG: Here should be new import; probably needs new handling
    train_data = load_tfrecord_data('data/raw/train_subset1.tfrecord')

    # new approach with dvc-tracking of data
    class_labels = []
    # decoding one-hot back to class indices
    for _, label in train_data:
        class_labels.append(tf.argmax(label, axis=-1).numpy()[0]) 

    return class_labels

def preprocess_image(file): # nvd06: changed from image_path to file to fit the main.yp and FastAPI setup. 
    '''
    Loads and preprocesses a single image for model prediction.

    Args:
    - img_path (str): Path to the image.

    Returns:
    - img_array: Preprocessed image array.
    '''
    img = image.load_img(BytesIO(file.file.read()), target_size=IMG_SIZE) # added the BytesIO part nvd06
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_single_image(file): # nvd06: changed from image_path to file to fit the main.yp and FastAPI setup.
    '''
    Predicts the class of a single image.

    Args:
    - img_path (str): Path to the image.

    Returns:
    - Predicted class label.
    '''
    model = load_trained_model()
    class_labels = get_class_labels()  # Get dynamic class labels
    img_array = preprocess_image(file) # nvd06: changed from image_path to file to fit the main.yp and FastAPI setup.
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]

    return class_labels[class_index]  # Return class label instead of index

if __name__ == '__main__':
    # Run predictions on the test folder by default
    print(f'Running predictions on the test set: {TEST_PATH}')
    # DEBUG: NEEDS UPDATE TO JUST SINGLE PICTURE
    results = predict_folder(TEST_PATH)

    print('\nPredictions for Test Set:')
    for filename, label in results.items():
        print(f'{filename}: {label}')

