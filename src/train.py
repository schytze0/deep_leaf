import tensorflow as tf
from tensorflow.keras import optimizers, callbacks # type: ignore
import json
from app.model import build_vgg16_model
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

# Debugging: Print environment variables to verify they're loaded
dagshub_username = os.getenv('DAGSHUB_USERNAME')
dagshub_key = os.getenv('DAGSHUB_KEY')

if not dagshub_username:
    print("❌ ERROR: DAGSHUB_USERNAME is not set.")
if not dagshub_key:
    print("❌ ERROR: DAGSHUB_KEY is not set.")

# Set environment variables
os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username or ""
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_key or ""

# ML Flow setup (still needs to be tested)
class MLFlowLogger(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_accuracy = 0
        self.best_epoch = 0
        self.best_run_id = None

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metric('train_loss', logs.get('loss'), step=epoch)
            mlflow.log_metric('train_accuracy', logs.get('accuracy'), step=epoch)
            mlflow.log_metric('val_loss', logs.get('val_loss'), step=epoch)
            mlflow.log_metric('val_accuracy', logs.get('val_accuracy'), step=epoch)

            # checking if it is the best epoch based on validation
            val_accuracy = logs.get('val_accuracy')
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_epoch = epoch
                self.best_run_id = mlflow.active_run().info.run_id

    def on_train_end(self, logs=None):
        if self.best_run_id is None:
            previous_best_run = mlflow.search_runs(order_by=['val_accuracy desc']).head(1)

            previous_best_run = mlflow.search_runs(order_by=['val_accuracy desc']).head(1)
            if previous_best_run is not None and not previous_best_run.empty:
                previous_best_val_accuracy = previous_best_run.iloc[0]['val_accuracy']
                if self.best_val_accuracy > previous_best_val_accuracy:
                    mlflow.log_param('best_epoch', self.best_epoch)
                    mlflow.log_metric('best_val_accuracy', self.best_val_accuracy)
                    self.best_run_id = mlflow.active_run().info.run_id
                    mlflow.log_param('best_run_id', self.best_run_id)
                    # Save the model again as it's the best so far
                    self.model.save(MODEL_PATH, save_format='keras')

def setup_mlflow_experiment():
    # TODO: set up later after Yannick created dagshub
    # DEBUG: Yannik, here you have to add the repository name and give each of us access to the repo via the API
    mlflow.set_tracking_uri('https://dagshub.com/schytze0/deep_leaf.mlflow')
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

# TODO: Created function to get best epoch (accuracy):
# DEBUG: Erwin, might already work, this was my first approach. Saving the new model is above after each epoch
def get_best_epoch_and_accuracy():
    # Fetching the best run from MLflow
    best_run = mlflow.search_runs(order_by=["val_accuracy desc"]).head(1)
    if best_run is not None and not best_run.empty:
        best_epoch = best_run.iloc[0]['best_epoch']
        best_val_accuracy = best_run.iloc[0]['best_val_accuracy']

        # log the values
        mlflow.log_metric('best_val_accuracy', best_val_accuracy)
        mlflow.log_param('best_epoch', best_epoch)

        # log run id
        run_id = best_run.iloc[0]['run_id']
        mlflow.log_param('best_run_id', run_id)

        return best_epoch, best_val_accuracy, run_id
    else:
        return None, None

# Old function adjusted
def train_model():
    '''
    Trains the model in two phases:
    1. Train only the classification head (with frozen base layers).
    2. Fine-tune the top layers of the base model with a smaller learning rate.
    3. Integrates MLflow to track scores (helpful if different training data is used; NOT TESTED YET)
    '''
        
    # load mlflow
    setup_mlflow_experiment()
    
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
        loss='categorical_crossentropy',
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
    mlflow_logger = MLFlowLogger()
    print('MLflow logger started ✅')

    # manually setting steps per epoch
    # steps_per_epoch = train_records // BATCH_SIZE

    print('Training classification head...', end='\r')
    history_1 = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=int(EPOCHS*0.7), 
        # steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint, mlflow_logger]
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
        # steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint, mlflow_logger]
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