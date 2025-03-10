import os
import mlflow
import mlflow.keras
import shutil
import subprocess
from dotenv import load_dotenv
from pathlib import Path
import tenacity

# imports from config
from config import MLFLOW_TRACKING_URL, MLFLOW_EXPERIMENT_NAME, DAGSHUB_REPO, MODEL_DVC

# Ensure script runs from this file path (or any of yours)
# REVIEW: Saved clone directory into .env, since it is for each of us different
os.chdir(os.getenv("CLONE_DIR"))  # If main branch clone is elsewhere (e.g., /home/olaf_wauzi/deep_leaf_main), update this path

def load_environment():
    """
    Load environment variables from .env file
    
    Returns:
    - username: DAGSHUB_USERNAME from .env
    - token: DAGSHUB_TOKEN from .env
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, "..", ".env")
    
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path, override=True)
        print('.env file found and loaded ✅')
    else:
        raise FileNotFoundError("Warning: .env file not found!")
    
    username = os.getenv('DAGSHUB_USERNAME')
    token = os.getenv('DAGSHUB_KEY')
    
    if not username or not token:
        raise ValueError("DAGSHUB_USERNAME or DAGSHUB_KEY not set in .env")
    
    print(f"Loaded credentials - Username: {username}")
    return username, token

def get_best_model():
    """
    Retrieves the best model from MLFlow based on validation accuracy.
    
    Returns:
        tuple: (model_path, best_val_accuracy, run_id)
    """
    # REVIEW: added both strings into config.py for faster change
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # REVIEW: I think we should compare the last val_accuracy which represents the last value of the fine-tuned model. 
    best_run = mlflow.search_runs(order_by=["metrics.final_val_accuracy DESC"]).head(1)
    if best_run.empty:
        print("No runs found in the experiment")
        return None, None, None
    
    run_id = best_run.iloc[0]['run_id']
    best_val_accuracy = best_run.iloc[0]['metrics.final_val_accuracy']
    # REVIEW: I changed to final accuracy value since we run two models and i think it should represent the modified model, shouldn't it?
    print(f"Run ID: {run_id}, Validation accuracy: {best_val_accuracy:.4f}")

    model_uri = f"runs:/{run_id}/model"
    # REVIEW: Maybe saving to an ignored temp-folder within the project? (I could try if you agree)
    local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    print(f"Model downloaded to: {local_model_path}")
    # Debug: List files to confirm structure
    artifact_files = list(Path(local_model_path).rglob("*"))
    print(f"Artifact files: {artifact_files}")
    return local_model_path, best_val_accuracy, run_id

def check_metadata_exists():
    """Check if metadata.txt exists in repo root and get its accuracy."""
    repo_root = Path.cwd()
    # REVIEW: Changed to save metadata also under models/
    metadata_path = repo_root / "models/metadata.txt"
    # REVIEW: changed to the specific file 
    dvc_pointer_path = repo_root / f'models/{MODEL_DVC}'
    
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            try:
                accuracy = float(f.read().strip())
                print(f"Existing metadata found with accuracy: {accuracy:.4f}")
                return True, accuracy
            except ValueError as e:
                print(f"Malformed metadata.txt: {e}. Assuming accuracy 0.0")
                return True, 0.0
    elif dvc_pointer_path.exists():
        print(f"Found {MODEL_DVC} but no metadata.txt; assuming accuracy 0.0")
        return True, 0.0
    else:
        print(f"No metadata.txt or {MODEL_DVC} found; assuming initial accuracy of 0.0")
        return False, 0.0

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(3))
def dvc_push(repo_root):
    """Retry DVC push up to 3 times with exponential backoff."""
    subprocess.run(["dvc", "push"], cwd=repo_root, check=True)
    print("Pushed model to DVC remote")

def upload_model_to_dagshub(username, token, model_artifact_path, val_accuracy, branch="main"): 
    """
    Copy the best model file into src/main/models/, create a DVC pointer file (models.dvc) and metadata.txt in the repo root, then commit and push changes.
    """
    repo_root = Path.cwd()
    # REVIEW: changed to folder models directly, the change via branches is made by git branch checkout.
    target_dir = repo_root / "models"  
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Directly fetch model.keras from the data/ subdirectory
    source_model_file = Path(model_artifact_path) / "data" / "model.keras"
    if not source_model_file.exists():
        artifact_files = list(Path(model_artifact_path).rglob("*"))
        print(f"Error: No model.keras found at {source_model_file}. Available files: {artifact_files}")
        raise FileNotFoundError(f"No model.keras found in {model_artifact_path}/data")
    
    dest_model_file = target_dir / "production_model.keras"
    shutil.copy2(source_model_file, dest_model_file)
    print(f"Copied model to {dest_model_file}")
    
    # Track with DVC
    # REVIEW: I needed to add str() here
    subprocess.run(["dvc", "add", str(dest_model_file)], cwd=repo_root, check=True)
    # REVIEW: This part did not work for me, with uncommenting it worked; added MODEL_DVC to config.py
    # dvc_file = target_dir / MODEL_DVC
    # target_dvc_file = repo_root / "models.dvc"
    # shutil.move(dvc_file, target_dvc_file)
    # print(f"Moved DVC pointer to {target_dvc_file}")
    dvc_file = target_dir / MODEL_DVC
    target_dvc_file = repo_root / MODEL_DVC
    shutil.move(dvc_file, target_dvc_file)
    print(f"Moved DVC pointer to {target_dvc_file}")
    
    # Update metadata.txt in repo root
    # REVIEW: Changed location of metadata.txt into models
    metadata_path = repo_root / "models/metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(str(val_accuracy))
    print(f"Updated {metadata_path} with accuracy: {val_accuracy:.4f}")
    
    # Git commit and push
    subprocess.run(["git", "add", str(target_dvc_file), str(metadata_path)], cwd=repo_root, check=True)
    commit_msg = f"Update model with accuracy {val_accuracy:.4f}"
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_root, check=True)
    subprocess.run(["git", "push", "origin", branch], cwd=repo_root, check=True)
    print("Pushed Git changes")
    
    # Configure DVC remote and push with retry
    dvc_remote = "origin"
    # REVIEW: Added variable from config instead of the string (DAGSHUB_REPO)
    subprocess.run(
        ["dvc", "remote", "add", "-f", "-d", dvc_remote, DAGSHUB_REPO, ".dvc"],
        cwd=repo_root, check=True
        )
    subprocess.run(["dvc", "remote", "modify", dvc_remote, "--local", "auth", "basic"], cwd=repo_root, check=True)
    subprocess.run(["dvc", "remote", "modify", dvc_remote, "--local", "user", username], cwd=repo_root, check=True)
    subprocess.run(["dvc", "remote", "modify", dvc_remote, "--local", "password", token], cwd=repo_root, check=True)
    try:
        dvc_push(repo_root)
    except Exception as e:
        print(f"❌ DVC push failed after retries: {e}")
        raise
    
    return True

def manual_upload_instructions(username, model_path, val_accuracy):
    print("\n=== MANUAL UPLOAD INSTRUCTIONS ===")
    print(f"1. Locate model.keras in: {model_path}/data")
    print("2. Rename to 'plant_disease_model.keras' and move to 'models/'")  # If main branch uses src/models/, update to 'src/models/'.
    print("3. Run: dvc add models/production_model.keras")  # If main branch uses src/models/, update to 'dvc add src/models/plant_disease_model.keras'.
    print(f"4. Create/update models/metadata.txt with: {val_accuracy}")
    print("5. Run: git add models.dvc models/metadata.txt")
    print("6. Run: git commit -m 'Update model' && git push origin main")  # Change "dev-erwin" to "main" after merging into main branch.
    print("7. Run: dvc push")
    print("===============================\n")

def update_model_if_better():
    username, token = load_environment()
    best_model_path, best_val_accuracy, run_id = get_best_model()
    
    if not best_model_path:
        return "No models found in MLFlow experiments"
    
    exists, existing_accuracy = check_metadata_exists()
    action = "No change"
    
    if not exists or best_val_accuracy > existing_accuracy:
        action = "Created" if not exists else "Updated"
        try:
            upload_model_to_dagshub(username, token, best_model_path, best_val_accuracy)
            return f"Model {action.lower()} - New accuracy: {best_val_accuracy:.4f}" + (f", Previous: {existing_accuracy:.4f}" if exists else "")
        except Exception as e:
            print(f"Error during upload: {e}")
            manual_upload_instructions(username, best_model_path, best_val_accuracy)
            return f"Failed to {action.lower()} model; see manual instructions"
    return f"No change - Existing accuracy ({existing_accuracy:.4f}) >= Current ({best_val_accuracy:.4f})"

if __name__ == "__main__":
    result = update_model_if_better()
    print(f"Model management result: {result}")