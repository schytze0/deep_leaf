import tensorflow as tf
from tensorflow.keras import optimizers, callbacks  
import json
import os
import mlflow
import mlflow.keras
import subprocess 
from pathlib import Path
import psutil

# imports from other scripts
from src.config import HISTORY_PATH, EPOCHS, BATCH_SIZE, NUM_CLASSES, MLFLOW_TRACKING_URL, MLFLOW_EXPERIMENT_NAME
from src.model import build_vgg16_model
from src.helpers import load_tfrecord_data
from src.prod_model_select_mlflow_dagshub import update_model_if_better

# new class F1-Score
# F1 score to get better reporting
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

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

#################### Changes proposed to run ML Flow in a Docker container ##################
#################### CHANGE 1: Remove .env loading and use Docker environment variables #####
### Removed: Loading .env file and manual DagsHub env vars ###
### Reason: We pass these via docker-compose environment variables ###

## I'll keep the original code commented out for reference; using '##' in the beginning of each line
## Access dagshub 
## Load environment variables from .env file
## script_dir = os.path.dirname(os.path.abspath(__file__))
## dotenv_path = os.path.join(script_dir, "..", ".env")

## if os.path.exists(dotenv_path):
##     load_dotenv(dotenv_path, override=True)
##     print('.env file found and loaded ✅')
## else:
##     print("Warning: .env file not found!")

## Debugging: Print environment variables to verify they're loaded
## dagshub_username = os.getenv('DAGSHUB_USERNAME')
## dagshub_key = os.getenv('DAGSHUB_KEY')

## if not dagshub_username:
##     print("❌ ERROR: DAGSHUB_USERNAME is not set.")
## if not dagshub_key:
##     print("❌ ERROR: DAGSHUB_KEY is not set.")
###################################################################################

# ML Flow setup (still needs to be tested)
class MLFlowLogger(callbacks.Callback):
    def __init__(self, run_id=None):
        super().__init__()
        self.run_id = run_id  # save the runID
        self.best_val_accuracy = 0.0
        self.final_val_accuracy = 0.0
        self.final_val_f1_score = 0.0
        self.best_epoch = 0
        # DEBUG(Phil): doubled therefore, outcommented
        # self.final_val_accuracy = 0.0
        # self.final_val_f1_score = 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:
            mlflow.log_metric('train_loss', logs.get('loss', 0.0), step=epoch)
            mlflow.log_metric('train_accuracy', logs.get('accuracy', 0.0), step=epoch)
            mlflow.log_metric('train_f1_score', logs.get('f1_score', 0.0), step=epoch)
            mlflow.log_metric('val_loss', logs.get('val_loss', 0.0), step=epoch)
            mlflow.log_metric('val_accuracy', logs.get('val_accuracy', 0.0), step=epoch)
            mlflow.log_metric('val_f1_score', logs.get('val_f1_score', 0.0), step=epoch)

            val_accuracy = logs.get('val_accuracy', 0.0)
            val_f1_score = logs.get('val_f1_score', 0.0)

            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_val_f1_score = val_f1_score
                self.best_epoch = epoch
                mlflow.log_metric('best_val_accuracy', self.best_val_accuracy)
                mlflow.log_metric('best_val_f1_score', self.best_val_f1_score)
                mlflow.log_metric('best_epoch', self.best_epoch)
                print(f'Updated best validation accuracy: {self.best_val_accuracy:.4f} ✅')
        except Exception as e:
            print(f"MLflow logging error: {e}")

    def on_train_end(self, logs=None):
        logs = logs or {}
        try:
            # DEBUG(Phil): you don't need self.final_val_accuracy twice
            # self.final_val_accuracy = logs.get('val_accuracy', self.final_val_accuracy)
            # self.final_val_f1_score = logs.get('val_f1_score', self.final_val_f1_score)
            self.final_val_accuracy = logs.get("val_accuracy")
            self.final_val_f1_score = logs.get("val_f1_score")
            mlflow.log_metric('final_val_accuracy', self.final_val_accuracy)
            mlflow.log_metric('final_val_f1_score', self.final_val_f1_score)
            print(f'Best Validation Accuracy: {self.best_val_accuracy:.4f}')
            print(f'Final Validation Accuracy: {self.final_val_accuracy:.4f}')
        except Exception as e:
            print(f"MLflow logging error in on_train_end: {e}")

def setup_mlflow_experiment():
    ###################### Change 2: Configure MLflow with Dockerized server and DagsHub ##########
    ###############################################################################################
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    ###################### add debug message #####################
    print(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URL}")
    ##############################################################
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    ###################### add debug message #####################
    print(f"MLflow experiment set to: {MLFLOW_EXPERIMENT_NAME}")
    ##############################################################

    ###### moved metrics to 'train_model' as I'm getting errors of empty runs ######
    # parameters for logging
    ## mlflow.log_param('model', 'VGG16')
    ## mlflow.log_param('epochs', EPOCHS)
    ## mlflow.log_param('batch_size', BATCH_SIZE)
    ## mlflow.log_param('num_classes', NUM_CLASSES)
    ## mlflow.log_param('input_shape', str((224, 224, 3)))

    #### do we need the baseline for the metrics? These values are overwritten by the MLflow Logger ####
    # Final metrics
    ## mlflow.log_metric('final_val_accuracy', 0)
    ## mlflow.log_metric('final_val_f1_score', 0)

    # Best metrics
    ## mlflow.log_metric('best_val_accuracy', 0)
    ## mlflow.log_metric('best_val_f1_score', 0)
    ## mlflow.log_metric('best_epoch', 0)
    
    # Epoch metrics
    ## mlflow.log_metric('train_loss', 0, step=0)
    ## mlflow.log_metric('train_accuracy', 0, step=0)
    ## mlflow.log_metric('train_f1_score', 0, step=0)
    ## mlflow.log_metric('val_accuracy', 0, step=0)
    ## mlflow.log_metric('val_loss', 0, step=0)
    ## mlflow.log_metric('val_f1_score', 0, step=0)


# MAIN FUNCTION FOR TRAINING
def train_model(dataset_path: str = 'data/raw/train_subset6.tfrecord'):
    '''
    Trains the model in two phases:
    1. Train only the classification head (with frozen base layers).
    2. Fine-tune the top layers of the base model with a smaller learning rate.
    3. Integrates MLflow to track scores (helpful if different training data is used; NOT TESTED YET)
    
    Arguments:
    - dataset_path: ???

    Returns: None   
    
    '''
    ############################## Change 3: explicitly start ML Flow run ########################
    
    # Ensure tracking URI is correctly set before doing anything else
    if mlflow.get_tracking_uri() != MLFLOW_TRACKING_URL:
        print(f"Warning: MLflow tracking URI mismatch. Resetting to {MLFLOW_TRACKING_URL}")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    
    # Add a basic connectivity check for MLflow server --> debug
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Extract hostname from MLFLOW_TRACKING_URL
        from urllib.parse import urlparse
        hostname = urlparse(MLFLOW_TRACKING_URL).hostname
        port = urlparse(MLFLOW_TRACKING_URL).port or 5001
        s.connect((hostname, port))
        s.close()
        print(f"MLflow server at {hostname}:{port} is reachable ✅")
    except Exception as e:
        print(f"WARN: Cannot connect to MLflow server: {e}")
        print("MLflow tracking may not work correctly.")
    ########################################################################
    
    mlflow_uri = mlflow.get_tracking_uri()  # debug
    print(f"Active MLflow tracking URI: {mlflow_uri}")  # debug
    
    # DEBUG(Phil): This should be add the beginning, because now you set it twice (above and here again)
    # setup mlflow experiment
    setup_mlflow_experiment()
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id  # debug
        print(f"Started MLflow run with ID: {run_id}")  # debug
        # parameters for logging --> moved from setup_mlflow_experiment()
        mlflow.log_param('dataset', dataset_path)
        mlflow.log_param('model', 'VGG16')
        mlflow.log_param('epochs', EPOCHS)
        mlflow.log_param('batch_size', BATCH_SIZE)
        mlflow.log_param('num_classes', NUM_CLASSES)
        mlflow.log_param('input_shape', str((224, 224, 3)))
        print("Parameters logged successfully")  # debug

        # INFO(Phil): The metrics are not initiliased! Myabe they are in mlflfowLogger!

        # new insertion
        # DEBUG(Phil): Correct dataloading
        train_data, _ = load_tfrecord_data('data/raw/train_subset6.tfrecord')
        print('Training data loaded ✅')

        val_data, _ = load_tfrecord_data('data/raw/valid_subset6.tfrecord')
        print('Validation data loaded ✅')

        input_shape = (224, 224, 3)

        num_classes = NUM_CLASSES
    
        # Step 1: Train classification head with frozen base model
        model, _ = build_vgg16_model(input_shape, num_classes, trainable_base=False)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', F1Score(name='f1_score')]
        )
        print('Model built ✅')

        # logging in mlflow
        # INFO: Starting MLflow
        mlflow_logger = MLFlowLogger()
        print('MLflow logger started ✅')
        
        ####################### debug before model fit1 ################
        print("Testing MLflow logging before model.fit1...")
        try:
            # DEBUG(Phil): What is the intent for this? the value is not really descriptive
            mlflow.log_param("test_before_training", "value")
            print("MLflow logging before training successful ✅")
        except Exception as e:
            print(f"MLflow logging before training failed: {e}")
        ##############################################

        print('Training classification head...', end='\r')
        history_1 = model.fit(
            train_data, 
            validation_data=val_data, 
            epochs=int(EPOCHS*0.7), 
            callbacks=[mlflow_logger]
        )
        
        ####################### debug after model fit1 ################
        print("Testing MLflow logging after model.fit1...")
        try:
            # DEBUG(Phil): What is the intent for this? the value is not really descriptive
            mlflow.log_param("test_after_training", "value")
            print("MLflow logging after training successful ✅")
        except Exception as e:
            print(f"MLflow logging after training failed: {e}")
        ##############################################
        
        print('Training classification ended ✅')

         # Step 2: Fine-tune the last layers of the base model
        print('Build fine-tuning model...', end='\r')
        
        tf.keras.backend.clear_session()

        # reinitializing optimizer
        optimizer = optimizers.Adam(learning_rate=1e-4, amsgrad=True)
    
        model, _ = build_vgg16_model(
            input_shape, 
            num_classes, 
            trainable_base=True, 
            fine_tune_layers=4
        )

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', F1Score()]
        )

        print('Fine-tuning model built ✅')

        ####################### debug before model fit2 ################
        print("Testing MLflow logging before model.fit2...")
        try:
            # DEBUG(Phil): What is the intent for this? the value is not really descriptive
            mlflow.log_param("test_before_training", "value")
            print("MLflow logging before training successful ✅")
        except Exception as e:
            print(f"MLflow logging before training failed: {e}")
        ##############################################

        print('Fine-tuning head...', end='\r')
        history_2 = model.fit(
            train_data, 
            validation_data=val_data, 
            epochs=int(EPOCHS*0.3), 
            callbacks=[mlflow_logger]
        )
        
        ####################### debug after model fit2 ################
        print("Testing MLflow logging after model.fit2...")
        try:
            # DEBUG(Phil): What is the intent for this? the value is not really descriptive
            mlflow.log_param("test_after_training", "value")
            print("MLflow logging after training successful ✅")
        except Exception as e:
            print(f"MLflow logging after training failed: {e}")
        ##############################################
        
        print('Fine-tuning ended ✅')

        ################################ Change 4: remove git/dvc/dagshub registry; container is no repo  #######
        # Combine both training histories
        history = {
            'accuracy': history_1.history['accuracy'] + history_2.history['accuracy'],
            'f1_score': history_1.history['f1_score'] + history_2.history['f1_score'],
            'loss': history_1.history['loss'] + history_2.history['loss'],
            'val_accuracy': history_1.history['val_accuracy'] + history_2.history['val_accuracy'],
            'val_f1_score': history_1.history['val_f1_score'] + history_2.history['val_f1_score'],
            'val_loss': history_1.history['val_loss'] + history_2.history['val_loss']
        }
        
        # Log system metrics
        mlflow.log_metric('cpu_usage_percent', psutil.cpu_percent())
        mlflow.log_metric('memory_usage_percent', psutil.virtual_memory().percent)
        
        # Registering model on MLflow model registry
        mlflow.keras.log_model(model, 'model', registered_model_name="VGG16_Production")  # added registered_model_name
        ## add debug statement:
        print('Model logged to MLflow with registered name "VGG16_Production" ✅')
    
        # Log artifacts
        os.makedirs('temp', exist_ok=True)
        model.save('temp/current_model.keras')
        print('Current model saved under temp/current_model.keras ✅')
        mlflow.log_artifact('temp/current_model.keras', artifact_path='model')

        # write current val_accuracy to current_accuracy.txt
        with open('temp/current_accuracy.txt', 'w') as f:
            f.write(str(mlflow_logger.final_val_accuracy))
        print('Saved current final validation accuracy as temp/current_accuracy.txt ✅.')
        mlflow.log_artifact('temp/current_accuracy.txt')
        # Save history as JSON
        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f)
        mlflow.log_artifact(HISTORY_PATH)
        print(f'History saved in {HISTORY_PATH} ✅.')
        
        ################## part of change 3: removed; 'with' block automatically closes run ###########
        ## mlflow.end_run()
        ## print('Scores are saved with MLflow ✅.')

        # make git commit for history --> removed
        # Git commit and push
        ## repo_root = Path.cwd()
        ## subprocess.run(
        ##     ['git', 'add', str(HISTORY_PATH)], 
        ##     cwd=repo_root, 
        ##     check=True
        ## )
        ## commit_msg = 'Updated history logs'
        ## subprocess.run(
        ##     ['git', 'commit', '-m', commit_msg], 
        ##     cwd=repo_root, 
        ##     check=True
        ## )
        print('Training completed. ✅')

if __name__ == '__main__':
    train_model()
    # result = update_model_if_better()
    # print(f'Model management result: {result}')
