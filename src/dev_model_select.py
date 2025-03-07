import tensorflow as tf
# from tensorflow.keras import optimizers, callbacks
import json
from model import build_vgg16_model
from config import MODEL_PATH, HISTORY_PATH, EPOCHS, BATCH_SIZE, NUM_CLASSES
import os
from helpers import load_tfrecord_data
import mlflow
import mlflow.keras
from dotenv import load_dotenv

import shutil
import requests
from pathlib import Path
import git
import tempfile

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
    mlflow.set_tracking_uri('https://dagshub.com/schytze0/deep_leaf.mlflow')
    mlflow.set_experiment('Plant_Classification_Experiment')
    
    # Search for the best run based on 'validation accuracy'
    best_run = mlflow.search_runs(order_by = ["metrics.val_accuracy desc"]).head(1)
    
    # Provide comments for clarity
    if best_run.empty:
        print("No runs found in the experiment")
        return None, None, None
    
    # Extract information about the best run
    run_id = best_run.iloc[0]['run_id']
    best_val_accuracy = best_run.iloc[0]['metrics.val_accuracy']
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

def compare_and_update_model():
    """
    Compares the best model from MLFlow with the existing model on DagsHub.
    If the MLFlow model has better performance or if no model exists on DagsHub,
    updates/creates the model on DagsHub.
    
    Returns:
        str: Message indicating the action taken (created, overwritten, or no change)
    """
    # Get the best model from MLFlow
    best_model_path, best_val_accuracy, run_id = get_best_model()
    
    if best_model_path is None:
        return "No models found in MLFlow experiments"
    
    # Define the DagsHub repository and model path
    dagshub_repo = "schytze0/deep_leaf"
    dagshub_model_path = "src/main/models"  # TODO: confirm best model file name
    
    # Get DagsHub credentials from environment variables - already set globally in train.py
    username = os.environ.get('MLFLOW_TRACKING_USERNAME')
    token = os.environ.get('MLFLOW_TRACKING_PASSWORD')
    
    if not username or not token:
        return "DagsHub credentials not found in environment variables"
    
    # Check if the model exists on DagsHub
    model_exists = check_model_exists(username, token, dagshub_repo, dagshub_model_path)
    
    if model_exists:
        # Download the existing model from DagsHub
        existing_model_path = download_model_from_dagshub(username, token, dagshub_repo, dagshub_model_path)
        
        if existing_model_path:
            # Get the existing model's accuracy from metadata
            existing_val_accuracy = get_model_accuracy_from_metadata(existing_model_path)
            print(f"Existing model validation accuracy: {existing_val_accuracy:.4f}")
            
            # Compare performances
            if best_val_accuracy > existing_val_accuracy:
                # Update the model on DagsHub
                success = upload_model_to_dagshub(username, token, dagshub_repo, dagshub_model_path, best_model_path, best_val_accuracy)
                if success:
                    return f"Model overwritten - New accuracy: {best_val_accuracy:.4f}, Previous: {existing_val_accuracy:.4f}"
                else:
                    return f"Failed to upload new model. New accuracy: {best_val_accuracy:.4f}, Previous: {existing_val_accuracy:.4f}"
            else:
                return f"No change - Existing model ({existing_val_accuracy:.4f}) is better than new model ({best_val_accuracy:.4f})"
        else:
            # Failed to download but the model exists - try uploading anyway
            success = upload_model_to_dagshub(username, token, dagshub_repo, dagshub_model_path, best_model_path, best_val_accuracy)
            if success:
                return f"Model replaced - Could not download existing model. New accuracy: {best_val_accuracy:.4f}"
            else:
                return "Failed to upload new model and could not download existing model"
    else:
        # Create new model on DagsHub
        success = upload_model_to_dagshub(username, token, dagshub_repo, dagshub_model_path, best_model_path, best_val_accuracy)
        if success:
            return f"New model created with validation accuracy: {best_val_accuracy:.4f}"
        else:
            return "Failed to create new model on DagsHub"

def check_model_exists(username, token, repo, path):
    """
    Checks if a model exists at the specified path in the DagsHub repository
    
    Returns:
        bool: True if model exists, False otherwise
    """
    
    
    # url = f"https://api.dagshub.com/api/v1/repos/{repo}/contents/{path}"  # url for main branch
    url = f"https://api.dagshub.com/api/v1/repos/{repo}/contents/{path}?ref=dev-erwin"  # url for testing
    response = requests.get(url, auth=(username, token))
    
    if response.status_code != 200:
        return False
    contents = response.json()
    # Look for the production model file
    for item in contents:
        if item.get('name') == "plant_disease_model.keras":
            return True
    return False

def download_model_from_dagshub(username, token, repo, path):
    """
    Downloads a model from DagsHub using the DagsHub API
    
    Returns:
        str: Path to the downloaded model, or None if download failed
    """
    # Create a temporary directory to store the downloaded model
    temp_dir = os.path.join(os.getcwd(), "temp_model")
    os.makedirs(temp_dir, exist_ok=True)
    
    # First, list the files in the directory
    # api_url = f"https://api.dagshub.com/api/v1/repos/{repo}/contents/{path}"  # url for main branch
    api_url = f"https://api.dagshub.com/api/v1/repos/{repo}/contents/{path}?ref=dev-erwin"
    response = requests.get(api_url, auth=(username, token))
    
    if response.status_code != 200:
        print(f"Failed to list files in {path}: {response.status_code}")
        return None
    
    # Parse the response to get file information
    try:
        files_info = response.json()
        if not isinstance(files_info, list):
            files_info = [files_info]  # Handle single file response
        
        # Download each file in the directory
        for file_info in files_info:
            # Skip directories
            if file_info.get('type') == 'dir':
                continue
                
            file_name = file_info.get('name')
            download_url = file_info.get('download_url')
            if download_url:
                file_response = requests.get(download_url, auth=(username, token))
                if file_response.status_code == 200:
                    with open(os.path.join(temp_dir, file_name), 'wb') as f:
                        f.write(file_response.content)
                    print(f"Downloaded: {file_name}")
                else:
                    print(f"Failed to download {file_name}: {file_response.status_code}")
        # Verify that both the model file and metadata exist
        if "plant_disease_model.keras" in os.listdir(temp_dir):
            if "metadata.txt" not in os.listdir(temp_dir):
                with open(os.path.join(temp_dir, 'metadata.txt'), 'w') as f:
                    f.write('0.0')  # default value
            return temp_dir
        else:
            print("Model file 'plant_disease_model.keras' not found in the downloaded contents")
            return None
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return None

def upload_model_to_dagshub(username, token, repo, path, model_path, val_accuracy):
    """
    Uploads a model to DagsHub using Git operations
    
    Returns:
        bool: True if upload succeeded, False otherwise
    """
    try:
        # Create a temporary directory for git operations
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone the repository
            repo_url = f"https://{username}:{token}@github.com/{repo}.git"
            git_repo = git.Repo.clone_from(repo_url, temp_dir, branch="dev-erwin")  # TODO: remove , branch="dev-erwin" for main branch
            
            # Create the target directory if it doesn't exist
            target_dir = os.path.join(temp_dir, path)
            os.makedirs(target_dir, exist_ok=True)
            
            # Handle if the downloaded best model is a file:
            if os.path.isfile(model_path):
                shutil.copy2(model_path, os.path.join(target_dir, "plant_disease_model.keras"))
            else:
                # If it's a directory, try to locate the expected model file
                candidate_file = os.path.join(model_path, "plant_disease_model.keras")
                if os.path.exists(candidate_file):
                    shutil.copy2(candidate_file, os.path.join(target_dir, "plant_disease_model.keras"))
                else:
                    raise FileNotFoundError("The expected model file 'plant_disease_model.keras' was not found in the best model path.")
            
            # Save the validation accuracy as metadata with the model
            with open(os.path.join(target_dir, "metadata.txt"), "w") as f:
                f.write(str(val_accuracy))
            
            # Commit and push the changes
            git_repo.git.add(os.path.join(path, '*'))
            git_repo.git.commit('-m', f'Update model with validation accuracy {val_accuracy:.4f}')
            git_repo.git.push()
            
            print(f"Model successfully uploaded to DagsHub at {repo}/{path}")
            return True
            
    except Exception as e:
        print(f"Error uploading model: {str(e)}")
        print("Fallback message: To upload the model manually, please commit the files to your repository.")
        return False

def get_model_accuracy_from_metadata(model_path):
    """
    Extracts the model accuracy from the metadata file
    
    Returns:
        float: The accuracy value, or 0.0 if not found
    """
    try:
        metadata_path = os.path.join(model_path, "metadata.txt")
        with open(metadata_path, "r") as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0.0
    
def update_model_if_better():
    """
    Function combining functions above to compare and update model if better
    
    Returns:
        str: Result message from the compare_and_update_model function
    """
    result = compare_and_update_model()
    print(f"Model management result: {result}")
    return result

if __name__ == '__main__':
    update_model_if_better()