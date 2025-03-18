import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from io import BytesIO
import argparse

# user defined functions and libraries
from src.config import MODEL_PATH, IMG_SIZE, TEST_PATH
from src.helpers import load_tfrecord_data

# user defined class for f1 score
@tf.keras.utils.register_keras_serializable()  # Register the custom class
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


def load_trained_model():
    '''
    Loads the trained model from disk.
    
    Returns:
    - model: The loaded Keras model.
    '''
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'F1Score': F1Score}  # `custom_objects` instead of `custom_object`
    )

def get_class_labels():
    '''
    Retrieves class labels dynamically from the training data.

    Returns:
    - class_labels (dict): Mapping from index to class label.
    '''

    train_data = load_tfrecord_data('data/training/train.tfrecord')

    # Print the structure of train_data to understand its type and contents
    dataset = train_data[0]

    # class_labels = []
    class_labels = set()
    # decoding one-hot back to class indices
    for images, labels in dataset.take(1):
        # class_indices = np.argmax(labels, axis=-1)
        # class_labels.update(class_indices)
        # new approach
        class_indices = np.argmax(labels, axis=-1)  # Assuming the labels are one-hot encoded
        unique_labels = np.unique(class_indices)
        class_labels.update(unique_labels)

    print(f'CLass labels: {class_labels}')
    return sorted(class_labels)

def preprocess_image(file): # nvd06: changed from image_path to file to fit the main.yp and FastAPI setup. 
    '''
    Loads and preprocesses a single image for model prediction.

    Args:
    - file (str): Path to the image.

    Returns:
    - img_array: Preprocessed image array.
    '''
    if isinstance(file, str):  # If it's a file path (for CLI)
        img = image.load_img(file, target_size=IMG_SIZE)
    else:  # If it's a file-like object (for FastAPI)
        img = image.load_img(BytesIO(file.file.read()), target_size=IMG_SIZE)

    # img = image.load_img(BytesIO(file.file.read()), target_size=IMG_SIZE) # added the BytesIO part for file handling nvd06
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def preprocess_image_from_path(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_single_image(file): 
    '''
    Predicts the class of a single image.

    Args:
    - file (str): Path to the image.

    Returns:
    - Predicted class label.
    '''
    model = load_trained_model()
    
    class_labels = get_class_labels()  # Get dynamic class labels
    
    img_array = preprocess_image(file) 
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]

    print(f"Class index: {class_index}")
    print(f"Class labels: {class_labels}")

    return class_labels[class_index]  # Return class label instead of index

if __name__ == '__main__':
    # added to pass argument within terminal
    parser = argparse.ArgumentParser(description='Predict image class using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    # Run predictions on the test folder by default
    print(f'Running predictions on the test set: {args.image_path}')
    results = predict_single_image(args.image_path)

    print(f'\nPredictions for Test Set: {results}')
    # for filename, label in results.items():
    #     print(f'{filename}: {label}')