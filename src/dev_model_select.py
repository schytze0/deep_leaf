import os
import mlflow
import mlflow.keras
import shutil
import requests
import subprocess
import git
import tempfile
from dotenv import load_dotenv
from pathlib import Path
import dvc

# Load environment variables
def load_environment():
    """Load environment variables from .env file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, "..", ".env")
    
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path, override=True)
        print('.env file found and loaded ✅')
    else:
        raise FileNotFoundError("Warning: .env file not found!")
    
    # Set MLFlow environment variables
    username = os.getenv('DAGSHUB_USERNAME')
    token = os.getenv('DAGSHUB_KEY')
    
    if not username or not token:
        raise ValueError("DAGSHUB_USERNAME or DAGSHUB_KEY not set in .env")
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    
    print(f"Loaded credentials - Username: {username}, Token: {token[:5]}...")
    
    return username, token

def get_best_model():
    """
    Retrieves the best model from MLFlow based on validation accuracy.
    
    Returns:
        tuple: (model_path, best_val_accuracy, run_id)
    """
    # Set the MLFlow tracking URI
    mlflow.set_tracking_uri('https://dagshub.com/schytze0/deep_leaf.mlflow')
    mlflow.set_experiment('Plant_Classification_Experiment')
    
    # Search for the best run
    best_run = mlflow.search_runs(order_by=["metrics.val_accuracy DESC"]).head(1)
    
    if best_run.empty:
        print("No runs found in the experiment")
        return None, None, None
    
    # Extract information
    run_id = best_run.iloc[0]['run_id']
    best_val_accuracy = best_run.iloc[0]['metrics.val_accuracy']
    
    print(f"Best run ID: {run_id}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    # Download model artifacts
    model_uri = f"runs:/{run_id}/model"
    local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    print(f"Model downloaded to: {local_model_path}")
    
    return local_model_path, best_val_accuracy, run_id

def check_metadata_exists(username, token, branch="dev-erwin"):
    """
    Check if metadata.txt exists in the repository and get the accuracy value
    Use DVC with 'dvc-origin' for testing to pull the model and Git for metadata.
    
    Returns:
        tuple: (exists (bool), accuracy (float))
    """
    repo = "schytze0/deep_leaf"
    repo_url = f"https://{username}:{token}@dagshub.com/{repo}.git"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(["git", "clone", "-b", branch, repo_url, temp_dir], check=True)
        print(f"Cloned {repo} (branch: {branch}) to check metadata")
        
        # Check for metadata.txt and .dvc file without pulling all DVC data
        metadata_path = os.path.join(temp_dir, "models", "metadata.txt")
        dvc_path = os.path.join(temp_dir, "models", "plant_disease_model.keras.dvc")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                try:
                    existing_accuracy = float(f.read().strip())
                    print(f"Existing metadata found with accuracy: {existing_accuracy:.4f}")
                    return True, existing_accuracy
                except ValueError:
                    print("Invalid accuracy in metadata.txt; treating as 0.0")
                    return True, 0.0
        elif os.path.exists(dvc_path):
            print("Found model .dvc but no metadata.txt; assuming accuracy 0.0")
            return True, 0.0
        else:
            print("No metadata.txt or model .dvc found; assuming initial accuracy of 0.0")
            return False, 0.0

def upload_model_to_dagshub(username, token, model_path, val_accuracy, branch="dev-erwin"):
    """
    Uploads the model to DagsHub using DVC with 'dvc-origin' for testing for the model 
    and Git for metadata.
    
    Returns:
        bool: Success status
    """
    repo = "schytze0/deep_leaf"
    repo_url = f"https://{username}:{token}@dagshub.com/{repo}.git"
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Created temporary directory: {temp_dir}")
            subprocess.run(["git", "clone", "-b", branch, repo_url, temp_dir], check=True)
            print(f"Cloned repository {repo} (branch: {branch})")
            
            subprocess.run(["git", "-C", temp_dir, "config", "user.name", "MLflow Model Updater"], check=True)
            subprocess.run(["git", "-C", temp_dir, "config", "user.email", f"{username}@users.noreply.dagshub.com"], check=True)
            
            dvc_remote = "dvc-origin"
            result = subprocess.run(["dvc", "remote", "list"], cwd=temp_dir, capture_output=True, text=True)
            if dvc_remote not in result.stdout:
                subprocess.run(["dvc", "remote", "add", "-d", dvc_remote, f"https://dagshub.com/{repo}.dvc"], cwd=temp_dir, check=True)
                print(f"Added test DVC remote: {dvc_remote}")
            
            subprocess.run(["dvc", "remote", "modify", dvc_remote, "--local", "auth", "basic"], cwd=temp_dir, check=True)
            subprocess.run(["dvc", "remote", "modify", dvc_remote, "--local", "user", username], cwd=temp_dir, check=True)
            subprocess.run(["dvc", "remote", "modify", dvc_remote, "--local", "password", token], cwd=temp_dir, check=True)
            
            target_dir = os.path.join(temp_dir, "models")
            os.makedirs(target_dir, exist_ok=True)
            
            source_model_file = os.path.join(model_path, "data", "model.keras")
            target_model_path = os.path.join(target_dir, "plant_disease_model.keras")
            if os.path.exists(source_model_file):
                shutil.copy2(source_model_file, target_model_path)
                print(f"Copied and renamed model to {target_model_path}")
            else:
                raise FileNotFoundError(f"Model file not found at {source_model_file}")
            
            # Track model with DVC, expect models.dvc in root
            subprocess.run(["dvc", "add", "models/plant_disease_model.keras"], cwd=temp_dir, check=True)
            dvc_file = "models.dvc"
            print(f"Tracked model with DVC, created {dvc_file} in root")
            
            # Move .dvc file to models/
            shutil.move(os.path.join(temp_dir, dvc_file), os.path.join(target_dir, dvc_file))
            print(f"Moved {dvc_file} to models/")
            
            # Create metadata.txt after DVC add
            metadata_path = os.path.join(target_dir, "metadata.txt")
            with open(metadata_path, "w") as f:
                f.write(str(val_accuracy))
            print(f"Created metadata.txt with accuracy: {val_accuracy:.4f}")
            
            # Force-add files to Git
            subprocess.run(["git", "-C", temp_dir, "add", "-f", f"models/{dvc_file}", "models/metadata.txt"], check=True)
            print("Force-added .dvc and metadata.txt to Git")
            
            commit_result = subprocess.run(["git", "-C", temp_dir, "commit", "-m", f"Update model with accuracy {val_accuracy:.4f}"], 
                                         capture_output=True, text=True)
            if commit_result.returncode == 0 or "nothing to commit" in commit_result.stdout:
                print("Git commit successful or no changes")
            else:
                print(f"Commit failed: {commit_result.stderr}")
                raise Exception("Git commit failed")
            
            subprocess.run(["git", "-C", temp_dir, "push", repo_url, branch], check=True)
            print("Pushed Git changes (code and metadata) to DagsHub")
            
            subprocess.run(["dvc", "push"], cwd=temp_dir, check=True)
            print("Pushed model data to DVC remote")
            
            return True
    
    except Exception as e:
        print(f"Error uploading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def manual_upload_instructions(username, model_path, val_accuracy):
    """
    Provides manual upload instructions if automatic methods fail
    """
    print("\n=== MANUAL UPLOAD INSTRUCTIONS ===")
    print("Since automatic upload failed, upload the files manually:")
    print(f"1. Source model file: {os.path.join(model_path, 'data', 'model.keras')}")
    print("2. Rename to 'plant_disease_model.keras' and place in 'models/'")
    print("3. Run: dvc add models/plant_disease_model.keras")
    print(f"4. Create/update metadata.txt with: {val_accuracy}")
    print("5. Run: git add models/plant_disease_model.keras.dvc models/metadata.txt")
    print("6. Run: git commit -m 'Update model' && git push origin dev-erwin")
    print("7. Run: dvc push")
    print("===============================\n")

def update_model_if_better():
    """
    Compare and update model if better based on metadata.txt
    
    Returns:
        str: Result message
    """
    # Load environment and get credentials
    username, token = load_environment()
    
    if not username or not token:
        return "DagsHub credentials not found in environment variables"
    
    # Get the best model from MLFlow
    best_model_path, best_val_accuracy, run_id = get_best_model()
    
    if best_model_path is None:
        return "No models found in MLFlow experiments"
    
    # Check if metadata exists and get existing accuracy
    metadata_exists, existing_accuracy = check_metadata_exists(username, token)
    
    # Decision making based on metadata comparison
    if not metadata_exists:
        # No existing metadata/model, upload the new one
        print("No existing model found. Creating new model...")
        success = upload_model_to_dagshub(username, token, best_model_path, best_val_accuracy)
        if success:
            return f"New model created with validation accuracy: {best_val_accuracy:.4f}"
        else:
            manual_upload_instructions(username, best_model_path, best_val_accuracy)
            return "Failed to create new model; manual instructions provided above."
    else:
        # Compare performances using metadata values
        print(f"Comparing accuracies - New: {best_val_accuracy:.4f}, Existing: {existing_accuracy:.4f}")
        if best_val_accuracy > existing_accuracy:
            print("New model is better. Updating...")
            success = upload_model_to_dagshub(username, token, best_model_path, best_val_accuracy)
            if success:
                return f"Model updated - New accuracy: {best_val_accuracy:.4f}, Previous: {existing_accuracy:.4f}"
            else:
                return f"Failed to upload new model. New accuracy: {best_val_accuracy:.4f}, Previous: {existing_accuracy:.4f}"
        else:
            return f"No change - Existing model ({existing_accuracy:.4f}) is better or equal to new model ({best_val_accuracy:.4f})"

if __name__ == '__main__':
    result = update_model_if_better()
    print(f"Model management result: {result}")