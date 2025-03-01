import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import json
from model import build_vgg16_model
from config import MODEL_PATH, HISTORY_PATH, EPOCHS, BATCH_SIZE, NUM_CLASSES
import os
from helpers import load_tfrecord_data
import mlflow
import mlflow.keras
from dotenv import load_dotenv

# Access dagshub 
# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, "..", ".env")

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path, override=True)
    print('.env file found and loaded ✅')
else:
    print("Warning: .env file not found!")

os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_KEY')

# ML Flow setup (still needs to be tested)
# class MLFlowLogger(callbacks.Callback):
#     def __init__(self):
#         super().__init__()
#         self.best_val_accuracy = 0
#         self.best_epoch = 0
#         self.best_run_id = None

#     def on_epoch_end(self, epoch, logs=None):
#         if logs:
#             mlflow.log_metric('train_loss', logs.get('loss'), step=epoch)
#             mlflow.log_metric('train_accuracy', logs.get('accuracy'), step=epoch)
#             mlflow.log_metric('val_loss', logs.get('val_loss'), step=epoch)
#             mlflow.log_metric('val_accuracy', logs.get('val_accuracy'), step=epoch)

#             # checking if it is the best epoch based on validation
#             val_accuracy = logs.get('val_accuracy')
#             if val_accuracy > self.best_val_accuracy:
#                 self.best_val_accuracy = val_accuracy
#                 self.best_epoch = epoch
#                 self.best_run_id = mlflow.active_run().info.run_id

#     def on_train_end(self, logs=None):
#         if self.best_run_id is None:
#             previous_best_run = mlflow.search_runs(order_by=['val_accuracy desc']).head(1)

#             previous_best_run = mlflow.search_runs(order_by=['val_accuracy desc']).head(1)
#             if previous_best_run is not None and not previous_best_run.empty:
#                 previous_best_val_accuracy = previous_best_run.iloc[0]['val_accuracy']
#                 if self.best_val_accuracy > previous_best_val_accuracy:
#                     mlflow.log_param('best_epoch', self.best_epoch)
#                     mlflow.log_metric('best_val_accuracy', self.best_val_accuracy)
#                     self.best_run_id = mlflow.active_run().info.run_id
#                     mlflow.log_param('best_run_id', self.best_run_id)
#                     # Save the model again as it's the best so far
#                     self.model.save(MODEL_PATH, save_format='keras')

def setup_mlflow_experiment():
    # TODO: set up later after Yannick created dagshub
    # DEBUG: Yannik, here you have to add the repository name and give each of us access to the repo via the API
    mlflow.set_tracking_uri('https://dagshub.com/philkleer/deepleap_mlops.mlflow')
    mlflow.set_experiment('Plant_Classification_Experiment')

    mlflow.start_run()

    # parameters for logging
    mlflow.log_param('model', 'VGG16')
    mlflow.log_param('epochs', EPOCHS)
    mlflow.log_param('batch_size', BATCH_SIZE)
    mlflow.log_param('num_classes', NUM_CLASSES)
    mlflow.log_param('input_shape', (224, 224, 3))
    mlflow.log_param('best_run_id', 0)
    mlflow.log_param('best_epoch', 0)
    mlflow.log_param('best_val_accuracy', 0)
    mlflow.log_metric('val_accuracy', 0, step=0)
    mlflow.log_metric('train_loss', 0, step=0)
    mlflow.log_metric('train_accuracy', 0, step=0)
    mlflow.log_metric('val_loss', 0, step=0)

# TODO: Function to load a best model based on 'val_accuracy' from MLFlow's artifact storage to a local directory:
# STATUS: not tested
# DEBUG: Erwin - this is a first attempt; to be discussed further;
# DEBUG: removed 'best_epoch'; we can choose the best model based on 'val_accuracy' and 'run_id' 
def get_best_model():
    """
    Retrieves the best-of-the-best" model from MLFlow based on validation accuracy.
    Each stored model has been stored in MLFlow based on best 'val_accuracy' epoch in train_model().
    
    Returns:
        model_path (str): Path to the best model within MLFlow
        run_id (str): ID of the best run
        best_val_accuracy (float): Validation accuracy of the best model in MLFlow experiments
    """
    # Set the MLFlow tracking URI --> not set globally
    mlflow.set_tracking_uri('https://dagshub.com/philkleer/deepleap_mlops.mlflow')
    mlflow.set_experiment('Plant_Classification_Experiment')
    
    # Search for the best run based on 'validation accuracy'
    best_run = mlflow.search_runs(order_by = ["val_accuracy desc"]).head(1)
    
    # Provide comments for clarity
    if best_run.empty:
        print("No runs found in the experiment")
        return None, None, None
    
    # Extract information about the best run
    run_id = best_run.iloc[0]['run_id']
    best_val_accuracy = best_run.iloc[0]['best_val_accuracy']
    # best_epoch = best_run.iloc[0]['best_epoch']  # TODO: do we need this information elsewhere?
    
    # Adding some comments again for clarity
    print(f"Best run ID: {run_id}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    # Define the model uri using the 'run_id'
    model_uri = f"runs:/{run_id}/model"
    
    # Download the model to a local path
    local_model_path = mlflow.artifacts.download_artifacts(artifact_uri = model_uri)
    
    # TODO: if instead we need to use the model immediately for inference, further training, etc.
    # we can instead use the following:
    # best_model = mlflow.keras.load_model(model_uri)
    # print("Best model is loaded")
    # return best_model, best_val_accuracy, run_id

    # Indicate clarity on the model path
    print(f"Model downloaded to: {local_model_path}")
    
    return local_model_path, best_val_accuracy, run_id

################################ production ###########################################
# TODO: Erwin - for production we can use this code to load the best model retreived by get_best_model()
#
# # Get the best model
# model_path, run_id, accuracy = get_best_model() 
# if model_path:
#    # Load the model
#    best_model = mlflow.keras.load_model(model_path)
#    
#    # Copy the model to the production repository or deployment location
#    production_path = "/path/to/production/model"  # path to be defined
#   # Use appropriate methods for model transfer: shutil.copy, git operations, etc.
#    
#    print(f"Best model (accuracy: {accuracy:.4f}) deployed to production")
# else:
#     print("Failed to retrieve the best model")
#########################################################################################

# Old function adjusted
def train_model():
    '''
    Trains the model in two phases:
    1. Train only the classification head (with frozen base layers).
    2. Fine-tune the top layers of the base model with a smaller learning rate.
    3. Integrates MLflow to track scores (helpful if different training data is used; NOT TESTED YET)
    '''
        
    # load mlflow
    # setup_mlflow_experiment()
    
    # new insertion
    # TODO: Probably this could be part of the api, the path to the training data?
    train_data, train_records = load_tfrecord_data('data/raw/train_subset1.tfrecord')
    print('Training data loaded ✅')

    val_data, val_records = load_tfrecord_data('data/raw/valid_subset1.tfrecord')
    print('Validation data loaded ✅')

    input_shape = (224, 224, 3)

    num_classes = NUM_CLASSES
    
    # Step 1: Train classification head with frozen base model
    model, _ = build_vgg16_model(input_shape, num_classes, trainable_base=False)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print('Model built ✅')

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_PATH, 
        save_best_only=True, 
        monitor='val_accuracy', 
        mode='max'
    )

    # logging in mlflow
    # INFO: Starting MLflow
    # mlflow_logger = MLFlowLogger()
    print('MLflow logger started ✅')

    # manually setting steps per epoch
    steps_per_epoch = train_records // BATCH_SIZE

    print('Training classification head...', end='\r')
    history_1 = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=int(EPOCHS*0.7), 
        steps_per_epoch=steps_per_epoch,
        # callbacks=[checkpoint, mlflow_logger]
    )
    print('Training classification ended ✅')


    # Step 2: Fine-tune the last layers of the base model
    print('Fine-tuning model...', end='\r')
    model, _ = build_vgg16_model(
        input_shape, 
        num_classes, 
        trainable_base=True, 
        fine_tune_layers=4
    )
    print('Fine-tuning model ended ✅')

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print('Training classification head...', end='\r')
    history_2 = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=int(EPOCHS*0.3), 
        steps_per_epoch=steps_per_epoch,
        # callbacks=[checkpoint, mlflow_logger]
    )
    print('Training classification ended ✅')

    # saving mlflow loggs
    mlflow.keras.log_model(model, 'model')
    mlflow.end_run()
    print('Scores are saved with MLflow ✅.')

    # Combine both training histories
    history = {
        'accuracy': history_1.history['accuracy'] + history_2.history['accuracy'],
        'val_accuracy': history_1.history['val_accuracy'] + history_2.history['val_accuracy'],
        'loss': history_1.history['loss'] + history_2.history['loss'],
        'val_loss': history_1.history['val_loss'] + history_2.history['val_loss']
    }

    # Save history as JSON
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)
        print(f'History saved in {HISTORY_PATH} ✅.')

    print(f'Training completed. Model saved at {MODEL_PATH} ✅')

if __name__ == '__main__':
    train_model()

